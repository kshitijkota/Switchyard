"""LLM diagnosis node — AGENT_BRIEF §8.

A single contained job: given a cohort's failure-code counts over a time window
plus a baseline comparison, output structured JSON
{"cause", "confidence", "evidence"}. It NEVER routes and never imports ground
truth. It can (and should) answer INSUFFICIENT_EVIDENCE.

Guarantees:
  - responses are cached by input hash, so evaluation is reproducible and cheap;
  - output is validated against diagnose.schema.OUTPUT_SCHEMA — on any parse or
    validation failure the node falls back to INSUFFICIENT_EVIDENCE and logs the
    incident (never a wrong assertion from a broken response).

Provider selection (see diagnose/evaluate.py): Gemini is primary; the Anthropic
path stays intact and selectable; OpenAI is a fallback only for a non-rate-limit
Gemini failure; and a deterministic offline statistical diagnoser keeps the
pipeline runnable when no key is present. A committed Gemini cache reproduces the
Gemini numbers with no key at all. Keys are read from the environment only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import jsonschema

from diagnose.schema import (
    CAUSE_AMBIGUOUS, CAUSE_CLASSES, CODE_MEANING, INSUFFICIENT, OUTPUT_SCHEMA,
    VALID_OUTPUTS,
)

_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(_ROOT, "diagnose", "cache")
INCIDENT_LOG = os.path.join(_ROOT, "diagnose", "cache", "incidents.log")
LLM_MODEL = os.environ.get("SWITCHYARD_DIAGNOSE_MODEL", "claude-opus-5")
PROMPT_VERSION = "v5-real-codes-ambiguous"   # bump to invalidate the cache when build_prompt or the code set changes
MIN_COHORT_FAILURES = 40   # sample-size guardrail: below this the diagnoser abstains


# --- Control exceptions (propagate to the run loop; NOT per-call fallbacks) ------
class RateLimitExhausted(Exception):
    """A 429 that survived backoff — STOP the run, do not switch providers."""


class ProviderUnavailable(Exception):
    """A non-rate-limit failure (bad key, model unavailable) — a fallback provider
    may be tried."""


class SpendTripwire(Exception):
    """Estimated third-party spend crossed a hard limit — STOP immediately."""


class CacheMiss(Exception):
    """Cache-only reproduce mode hit a cohort with no committed answer and no key —
    the cohort is skipped so a cold clone reproduces exactly the committed set."""


def load_env() -> None:
    """Best-effort load of the repo-root .env (never overrides real env vars).
    Keys are read from the environment only; nothing is printed or logged."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(_ROOT, ".env"), override=False)
    except Exception:
        pass


# --- Config resolution (never trust a shared LLM_MODEL — see .env.example) -------
def gemini_model() -> str:
    return os.environ.get("GEMINI_MODEL") or os.environ.get("LLM_MODEL_gemini") or "gemini-3.8-flash"


def openai_model() -> str:
    return os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL_openai") or "gpt-4o-mini"


def max_rpm() -> int:
    try:
        return int(os.environ.get("LLM_MAX_RPM", "60"))
    except ValueError:
        return 60


@dataclass
class DiagnosisInput:
    cohort_label: str
    window: str
    cohort_counts: dict          # code -> count in the cohort
    baseline_counts: dict        # code -> count in the comparison baseline

    def canonical(self) -> dict:
        return {
            "cohort_label": self.cohort_label,
            "window": self.window,
            "cohort_counts": {k: int(self.cohort_counts.get(k, 0)) for k in sorted(CODE_MEANING)},
            "baseline_counts": {k: int(self.baseline_counts.get(k, 0)) for k in sorted(CODE_MEANING)},
        }

    def hash(self) -> str:
        # Include the prompt version so a prompt change auto-invalidates the cache.
        blob = json.dumps({**self.canonical(), "_prompt": PROMPT_VERSION}, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:24]


def _fallback(reason: str) -> dict:
    return {"cause": INSUFFICIENT, "confidence": 0.0, "evidence": [f"fallback: {reason}"]}


def _log_incident(input_hash: str, reason: str, raw: str = "") -> None:
    os.makedirs(os.path.dirname(INCIDENT_LOG), exist_ok=True)
    with open(INCIDENT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"hash": input_hash, "reason": reason, "raw": raw[:500]}) + "\n")


def _validate(obj) -> dict | None:
    try:
        jsonschema.validate(obj, OUTPUT_SCHEMA)
        return obj
    except (jsonschema.ValidationError, TypeError):
        return None


def build_prompt(inp: "DiagnosisInput") -> str:
    """Shared prompt for every LLM provider (Anthropic/Gemini/OpenAI). Presents
    per-code SHARES (proportions) for cohort vs baseline, because the cohort is a
    small time-window sample and always has far fewer total failures than the
    all-time baseline — comparing raw counts makes weaker models wrongly conclude
    'nothing is elevated'."""
    cc = inp.canonical()["cohort_counts"]
    bc = inp.canonical()["baseline_counts"]
    n_c = sum(cc.values()) or 1
    n_b = sum(bc.values()) or 1
    rows = []
    for code in sorted(CODE_MEANING):
        meaning, cause = CODE_MEANING[code]
        rows.append(f"  {code} ({meaning} → {cause}): cohort {cc[code]/n_c:.0%} "
                    f"({cc[code]}/{n_c}) vs baseline {bc[code]/n_b:.0%}")
    table = "\n".join(rows)
    return (
        "You are a payments failure-diagnosis service. Given a cohort of failed "
        "transactions and an all-time baseline, decide the single most likely "
        "underlying CAUSE, or abstain.\n\n"
        f"Valid causes: {', '.join(VALID_OUTPUTS)}.\n\n"
        f"Cohort: {inp.cohort_label} over {inp.window}. Total cohort failures: {n_c}.\n"
        f"Failure-code share — cohort vs baseline (with the cause each code implies):\n{table}\n\n"
        "How to decide, in order:\n"
        "1. Sample size first. A cohort of only a few dozen total failures (roughly forty "
        "or fewer) is too small to attribute a cause reliably — return "
        "INSUFFICIENT_EVIDENCE.\n"
        "2. Otherwise read the cohort's failure-code SHARES (proportions, not raw counts). "
        "Group codes by the cause each implies and find the cause with the largest cohort "
        "share; the baseline column shows what is normal, so a cause standing well above "
        "its baseline share is a strong signal.\n"
        f"   Codes marked '→ {CAUSE_AMBIGUOUS}' name two possible causes in their own "
        "documentation and so identify NONE on their own; they are not a valid answer. If "
        "such codes dominate the cohort and no genuine cause clearly stands out beneath "
        "them, return INSUFFICIENT_EVIDENCE.\n"
        "3. Assert that one cause only if it is CLEARLY the dominant failure mode (well "
        "ahead of the others). If no cause clearly dominates, or two causes are comparably "
        "large, return INSUFFICIENT_EVIDENCE.\n"
        "Abstaining when genuinely unsure is correct and rewarded; a confident wrong "
        "assertion is not.\n\n"
        'Respond with ONLY a JSON object: {"cause": <one of the valid causes>, '
        '"confidence": <0..1>, "evidence": [<short strings>]}.'
    )


def _parse_json_lenient(text: str | None) -> dict | None:
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t[:4].lower() == "json":
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


# --- Providers ------------------------------------------------------------------

class StatisticalProvider:
    """Deterministic offline diagnoser. Compares the cohort's per-cause code share
    to baseline; asserts a processor/network/merchant cause only when it is
    materially elevated with enough samples, asserts CUSTOMER_SIDE only when it
    clearly dominates, and otherwise abstains. Uses ONLY public code meanings and
    the observed distributions — never the latent regime labels."""

    name = "statistical"
    MIN_SAMPLES = 40
    ELEV_THRESH = 0.15
    CUSTOMER_DOMINANCE = 0.55

    def _shares(self, counts: dict) -> dict:
        total = sum(counts.values()) or 1
        share = {c: 0.0 for c in CAUSE_CLASSES}
        for code, n in counts.items():
            cause = CODE_MEANING[code][1]
            if cause in share:
                share[cause] += n / total
            # else: ambiguous code (e.g. U30) — its mass is unattributable and
            # counts toward no cause, so a cohort dominated by it shows no elevated
            # cause and the provider abstains (INSUFFICIENT_EVIDENCE) below.
        return share

    def diagnose(self, inp: DiagnosisInput) -> dict:
        n = sum(inp.cohort_counts.values())
        if n < self.MIN_SAMPLES:
            return {"cause": INSUFFICIENT, "confidence": 0.3,
                    "evidence": [f"only {n} failures (< {self.MIN_SAMPLES}); too few to conclude"]}
        cohort = self._shares(inp.cohort_counts)
        base = self._shares(inp.baseline_counts)
        elevation = {c: cohort[c] - base[c] for c in CAUSE_CLASSES}
        # processor/network/merchant: look for a clearly elevated cause
        anomalous = {c: elevation[c] for c in ("ISSUER_DEGRADATION", "MERCHANT_INTEGRATION", "NETWORK_TRANSIENT")}
        top = max(anomalous, key=anomalous.get)
        if anomalous[top] >= self.ELEV_THRESH:
            return {"cause": top, "confidence": round(min(0.99, 0.5 + anomalous[top]), 2),
                    "evidence": [f"{top} code share {cohort[top]:.0%} vs baseline {base[top]:.0%} "
                                 f"(+{anomalous[top]:.0%})", f"{n} failures"]}
        if cohort["CUSTOMER_SIDE"] >= self.CUSTOMER_DOMINANCE and elevation["CUSTOMER_SIDE"] >= 0:
            return {"cause": "CUSTOMER_SIDE", "confidence": round(cohort["CUSTOMER_SIDE"], 2),
                    "evidence": [f"customer-side share {cohort['CUSTOMER_SIDE']:.0%} dominates; "
                                 f"no processor-side anomaly", f"{n} failures"]}
        return {"cause": INSUFFICIENT, "confidence": 0.4,
                "evidence": [f"no cause materially elevated over baseline (top {top} "
                             f"+{anomalous[top]:.0%})"]}


class LLMProvider:
    """Anthropic LLM diagnoser. Structured output only. Requires credentials + SDK
    (raises at construction otherwise). Kept intact and selectable."""

    name = "anthropic"

    def __init__(self, model: str = LLM_MODEL):
        import anthropic  # raises if SDK missing
        self.client = anthropic.Anthropic()  # raises later if no credentials
        self.model = model
        self.last_in = self.last_out = 0
        self.total_in = self.total_out = 0

    def diagnose(self, inp: DiagnosisInput) -> dict | None:
        try:
            resp = self.client.messages.create(
                model=self.model, max_tokens=1024,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": build_prompt(inp)}],
            )
        except Exception as e:  # noqa: BLE001
            raise ProviderUnavailable(f"anthropic: {type(e).__name__}: {e}")
        u = getattr(resp, "usage", None)
        self.last_in = int(getattr(u, "input_tokens", 0) or 0)
        self.last_out = int(getattr(u, "output_tokens", 0) or 0)
        self.total_in += self.last_in; self.total_out += self.last_out
        text = "".join(b.text for b in resp.content if b.type == "text")
        return _parse_json_lenient(text)


class GeminiProvider:
    """Primary provider (AGENT_BRIEF-final PROVIDER CONFIGURATION). Gemini flash,
    temperature 0, thinking disabled (five-way classification — thinking tokens
    would be billed for no benefit), strict JSON. Backs off on 429 and raises
    RateLimitExhausted if the quota is truly gone (the run then stops)."""

    name = "gemini"

    def __init__(self, model: str, api_key: str, rpm: int = 60):
        from google import genai
        from google.genai import errors, types
        self._types = types
        self._errors = errors
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._min_interval = 60.0 / max(1, rpm)
        self._last_ts = 0.0
        self._thinking_ok = True
        self.last_in = self.last_out = 0
        self.total_in = self.total_out = 0

    def _throttle(self) -> None:
        import time
        dt = time.time() - self._last_ts
        if dt < self._min_interval:
            time.sleep(self._min_interval - dt)
        self._last_ts = time.time()

    def diagnose(self, inp: DiagnosisInput) -> dict | None:
        import random
        import time
        prompt = build_prompt(inp)
        for attempt in range(6):
            self._throttle()
            try:
                cfg = dict(
                    temperature=0.0, response_mime_type="application/json",
                    automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
                )
                if self._thinking_ok:
                    cfg["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt,
                    config=self._types.GenerateContentConfig(**cfg))
            except self._errors.APIError as e:
                code = getattr(e, "code", None)
                msg = str(e).lower()
                if code in (500, 502, 503, 504):        # transient server — retry
                    time.sleep(min(60.0, 2 ** attempt) + random.uniform(0, 1))
                    continue
                if code == 429:
                    # A per-MINUTE cap resets quickly, so back off and retry. A
                    # per-DAY / project quota will NOT reset for hours — retrying
                    # only burns more of it, so stop immediately (partial run;
                    # the committed cache resumes next time). Never switch to OpenAI.
                    if "perminute" in msg or "per minute" in msg:
                        time.sleep(min(60.0, 2 ** attempt) + random.uniform(0, 1))
                        continue
                    raise RateLimitExhausted(f"gemini quota exhausted (per-day/project): {e}")
                if self._thinking_ok and "thinking" in msg:
                    self._thinking_ok = False
                    continue
                # Permanent (bad key / model unavailable / bad request) — a fallback
                # provider may be tried.
                raise ProviderUnavailable(f"gemini {code}: {e}")
            except Exception as e:  # noqa: BLE001
                raise ProviderUnavailable(f"gemini: {type(e).__name__}: {e}")
            um = resp.usage_metadata
            self.last_in = int(getattr(um, "prompt_token_count", 0) or 0)
            self.last_out = (int(getattr(um, "candidates_token_count", 0) or 0)
                             + int(getattr(um, "thoughts_token_count", 0) or 0))
            self.total_in += self.last_in; self.total_out += self.last_out
            return _parse_json_lenient(getattr(resp, "text", None))
        raise RateLimitExhausted("gemini unavailable (429/5xx) after 6 backoff attempts")


class OpenAIProvider:
    """OpenAI diagnoser (gpt-4o-mini). Used as the completing provider here after
    Gemini's free-tier daily quota was exhausted. Tracks estimated spend with a
    $5 runaway tripwire, well under the authorised $20 hard cap."""

    name = "openai"
    _PRICE = {"gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000)}
    # User raised the ceiling to $20 to complete the diagnosis. Expected real cost
    # for 19 tiny classifications is ~$0.01; the $5 tripwire is a runaway guard
    # (250× expected) well under the $20 hard cap.
    TRIPWIRE_USD = 5.0
    HARD_CAP_USD = 20.0

    def __init__(self, model: str, api_key: str):
        import openai
        self._openai = openai
        # Accept-Encoding: identity avoids a decompression bug in the installed
        # httpx2 build that otherwise raises a spurious connection error.
        self.client = openai.OpenAI(api_key=api_key, default_headers={"Accept-Encoding": "identity"})
        self.model = model
        self.last_in = self.last_out = 0
        self.total_in = self.total_out = 0
        self.spend_usd = 0.0

    def diagnose(self, inp: DiagnosisInput) -> dict | None:
        if self.spend_usd > self.TRIPWIRE_USD:
            raise SpendTripwire(f"estimated spend ${self.spend_usd:.4f} > ${self.TRIPWIRE_USD}")
        try:
            resp = self.client.chat.completions.create(
                model=self.model, temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": build_prompt(inp)}])
        except self._openai.RateLimitError as e:
            raise RateLimitExhausted(f"openai 429: {e}")
        except Exception as e:  # noqa: BLE001
            raise ProviderUnavailable(f"openai: {type(e).__name__}: {e}")
        u = resp.usage
        self.last_in = int(u.prompt_tokens); self.last_out = int(u.completion_tokens)
        self.total_in += self.last_in; self.total_out += self.last_out
        pin, pout = self._PRICE.get(self.model, (0.15 / 1e6, 0.60 / 1e6))
        self.spend_usd += self.last_in * pin + self.last_out * pout
        if self.spend_usd > self.HARD_CAP_USD:
            raise SpendTripwire(f"hard cap ${self.HARD_CAP_USD} reached")
        return _parse_json_lenient(resp.choices[0].message.content)


class Diagnoser:
    """Caches by (namespace, input hash). `provider=None` with an explicit
    cache_namespace is CACHE-ONLY reproduce mode — it serves committed answers
    without a key and fails over to INSUFFICIENT_EVIDENCE on a miss. Control
    exceptions (rate limit / provider unavailable / spend tripwire) PROPAGATE to
    the run loop; only malformed responses fall back per-call."""

    def __init__(self, provider=None, cache_dir: str = CACHE_DIR, cache_namespace: str | None = None):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        if provider is None and cache_namespace is None:
            provider = self._auto_provider()
        self.provider = provider
        self.cache_namespace = cache_namespace or (provider.name if provider else "unknown")
        self.parse_failures = 0
        self.cache_misses_without_provider = 0

    @staticmethod
    def _auto_provider():
        try:
            return LLMProvider()
        except Exception:
            return StatisticalProvider()

    def _cache_path(self, h: str) -> str:
        return os.path.join(self.cache_dir, f"{self.cache_namespace}_{h}.json")

    def diagnose(self, inp: DiagnosisInput, use_cache: bool = True) -> dict:
        # Sample-size guardrail FIRST: a production diagnoser must not attribute a
        # cause from a handful of failures. Abstain deterministically — no model
        # call, no cache needed — so it reproduces everywhere (incl. cold clone).
        n_cohort = sum(inp.cohort_counts.values())
        if n_cohort < MIN_COHORT_FAILURES:
            return {"cause": INSUFFICIENT, "confidence": 0.0, "_provider": self.cache_namespace,
                    "_tokens": [0, 0],
                    "evidence": [f"guardrail: {n_cohort} failures (< {MIN_COHORT_FAILURES}); "
                                 "too few to attribute a cause"]}

        h = inp.hash()
        path = self._cache_path(h)
        if use_cache and os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)

        if self.provider is None:  # cache-only reproduce mode, and a miss
            self.cache_misses_without_provider += 1
            raise CacheMiss(h)   # caller skips it; nothing is cached (no fake answer)
        else:
            raw = self.provider.diagnose(inp)   # control exceptions propagate
            result = _validate(raw)
            if result is None:
                self.parse_failures += 1
                _log_incident(h, "schema_validation_failed", json.dumps(raw) if raw else "")
                result = _fallback("schema_validation_failed")
            tokens = [int(getattr(self.provider, "last_in", 0)), int(getattr(self.provider, "last_out", 0))]

        result["_provider"] = self.cache_namespace
        result["_tokens"] = tokens
        with open(path, "w") as fh:
            json.dump(result, fh, sort_keys=True)
        return result
