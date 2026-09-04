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

Provider selection: the Anthropic LLM when a key + SDK are available, else a
deterministic offline statistical diagnoser so the pipeline is always runnable
and scoreable. Both obey the same contract.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import jsonschema

from diagnose.schema import (
    CAUSE_CLASSES, CODE_MEANING, INSUFFICIENT, OUTPUT_SCHEMA, VALID_OUTPUTS,
)

_ROOT = os.path.dirname(os.path.dirname(__file__))
CACHE_DIR = os.path.join(_ROOT, "diagnose", "cache")
INCIDENT_LOG = os.path.join(_ROOT, "diagnose", "cache", "incidents.log")
LLM_MODEL = os.environ.get("SWITCHYARD_DIAGNOSE_MODEL", "claude-opus-5")


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
        blob = json.dumps(self.canonical(), sort_keys=True).encode()
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
            share[CODE_MEANING[code][1]] += n / total
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
    (raises at construction otherwise, so the node falls back to statistical)."""

    name = "llm"

    def __init__(self, model: str = LLM_MODEL):
        import anthropic  # raises if SDK missing
        self.client = anthropic.Anthropic()  # raises later if no credentials
        self.model = model

    def _prompt(self, inp: DiagnosisInput) -> str:
        meanings = "\n".join(f"  {c}: {CODE_MEANING[c][0]}" for c in sorted(CODE_MEANING))
        return (
            "You are a payments failure-diagnosis service. Given a cohort of failed "
            "transactions (counts by failure code) and a baseline comparison, decide the "
            "single most likely underlying CAUSE, or abstain.\n\n"
            f"Failure code meanings:\n{meanings}\n\n"
            f"Valid causes: {', '.join(VALID_OUTPUTS)}.\n"
            "Return INSUFFICIENT_EVIDENCE when the cohort is small or no cause is clearly "
            "elevated over baseline — abstaining is correct and rewarded, a wrong assertion "
            "is not.\n\n"
            f"Cohort: {inp.cohort_label} over {inp.window}\n"
            f"Cohort failure counts: {json.dumps(inp.canonical()['cohort_counts'])}\n"
            f"Baseline failure counts: {json.dumps(inp.canonical()['baseline_counts'])}\n\n"
            'Respond with ONLY a JSON object: {"cause": <one of the valid causes>, '
            '"confidence": <0..1>, "evidence": [<short strings>]}.'
        )

    def diagnose(self, inp: DiagnosisInput) -> dict | None:
        resp = self.client.messages.create(
            model=self.model, max_tokens=1024,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": self._prompt(inp)}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        # tolerate code fences
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None  # node will validate/fallback and log


class Diagnoser:
    def __init__(self, provider=None, cache_dir: str = CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.provider = provider or self._auto_provider()

    @staticmethod
    def _auto_provider():
        try:
            return LLMProvider()
        except Exception:
            return StatisticalProvider()

    def _cache_path(self, h: str) -> str:
        # Namespace by provider so a statistical-provider cache never shadows the
        # LLM path (and vice versa) when a key becomes available.
        return os.path.join(self.cache_dir, f"{self.provider.name}_{h}.json")

    def diagnose(self, inp: DiagnosisInput, use_cache: bool = True) -> dict:
        h = inp.hash()
        path = self._cache_path(h)
        if use_cache and os.path.exists(path):
            with open(path) as fh:
                return json.load(fh)

        try:
            raw = self.provider.diagnose(inp)
        except Exception as e:  # any provider/API failure
            _log_incident(h, f"provider_error:{type(e).__name__}")
            raw = _fallback(f"provider_error:{type(e).__name__}")

        result = _validate(raw)
        if result is None:
            _log_incident(h, "schema_validation_failed", json.dumps(raw) if raw else "")
            result = _fallback("schema_validation_failed")

        result["_provider"] = self.provider.name
        with open(path, "w") as fh:
            json.dump(result, fh, sort_keys=True)
        return result
