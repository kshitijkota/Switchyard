"""Score the diagnosis node — AGENT_BRIEF §8 + final-task-list TASK 2.

Runs the cohorts against a REAL model (Gemini primary; OpenAI only on a
non-rate-limit Gemini failure; Anthropic selectable; offline statistical only
when no key is present). Reports, over the cohorts that completed:
  - accuracy on CLEAR cohorts (did it name the right cause?),
  - abstention rate on deliberately AMBIGUOUS cohorts (correctly said
    INSUFFICIENT_EVIDENCE?) — abstention is REWARDED, not penalised, because a
    calibrated "I don't know" is the correct answer for an un-diagnosable cohort
    and is strictly better than a confident wrong guess,
  - a harmful-error rate (a specific WRONG assertion),
  - parse-failure rate, total input/output tokens, and which provider/model
    produced the numbers.
Partial runs (quota exhausted mid-dataset) are labelled honestly.

Writes artifacts/diagnose_results.json.
"""

from __future__ import annotations

import glob
import json
import os

from diagnose.cohorts import build_cohorts
from diagnose.node import (
    CACHE_DIR, CacheMiss, Diagnoser, DiagnosisInput, GeminiProvider, LLMProvider,
    OpenAIProvider, ProviderUnavailable, RateLimitExhausted, SpendTripwire,
    StatisticalProvider, gemini_model, load_env, max_rpm, openai_model,
)
from diagnose.schema import INSUFFICIENT

_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(_ROOT, "artifacts", "diagnose_results.json")
STATISTICAL_REF_PATH = os.path.join(_ROOT, "artifacts", "diagnose_results_statistical.json")


def _committed_gemini_cache() -> bool:
    return bool(glob.glob(os.path.join(CACHE_DIR, "gemini_*.json")))


def _make_primary(force: str | None = None):
    """(diagnoser, label, model) — Gemini first; else committed-Gemini-cache
    reproduce (no key); else Anthropic; else offline statistical."""
    if force == "statistical":
        return Diagnoser(StatisticalProvider(), cache_namespace="statistical"), \
            "offline statistical (NOT a language model)", None
    gk = os.environ.get("GEMINI_API_KEY")
    if gk:
        return Diagnoser(GeminiProvider(gemini_model(), gk, rpm=max_rpm()),
                         cache_namespace="gemini"), "gemini (live)", gemini_model()
    if _committed_gemini_cache():
        return Diagnoser(None, cache_namespace="gemini"), "gemini (from committed cache)", gemini_model()
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            p = LLMProvider()
            return Diagnoser(p, cache_namespace="anthropic"), "anthropic (live)", p.model
        except Exception:
            pass
    return Diagnoser(StatisticalProvider(), cache_namespace="statistical"), \
        "offline statistical (NOT a language model)", None


def _run_cohorts(diagnoser, cohorts, baseline, use_cache):
    """Skip-and-continue on rate limits / cache misses so a run scores every cached
    cohort and only skips the ones it truly cannot get; a 3-consecutive-rate-limit
    circuit breaker bounds wasted time when the quota is exhausted. Returns
    (results, n_skipped, fatal)."""
    results, skipped, fatal, consec = [], 0, None, 0
    for co in cohorts:
        try:
            pred = diagnoser.diagnose(
                DiagnosisInput(co.label, co.window, co.counts, baseline), use_cache=use_cache)
        except CacheMiss:
            skipped += 1; continue                       # cold-clone reproduce: skip uncached
        except RateLimitExhausted:
            skipped += 1; consec += 1
            if consec >= 3:
                fatal = ("rate_limit_exhausted", "3 consecutive rate-limit skips"); break
            continue
        except SpendTripwire as e:
            fatal = ("spend_tripwire", str(e)); break
        except ProviderUnavailable as e:
            fatal = ("provider_unavailable", str(e)); break
        consec = 0
        results.append((co, pred))
    return results, skipped, fatal


def _is_fallback(pred: dict) -> bool:
    ev = pred.get("evidence") or []
    return bool(ev) and isinstance(ev[0], str) and ev[0].startswith("fallback:")


def evaluate(seed: int = 0, use_cache: bool = True, force_provider: str | None = None) -> dict:
    load_env()
    cohorts, baseline = build_cohorts(seed)
    diagnoser, label, model = _make_primary(force_provider)
    results, skipped, fatal = _run_cohorts(diagnoser, cohorts, baseline, use_cache)

    fallback_note = None
    # OpenAI fallback ONLY for a non-rate-limit *live Gemini* failure.
    if fatal and fatal[0] == "provider_unavailable" and label == "gemini (live)":
        ok = os.environ.get("OPENAI_API_KEY")
        if ok:
            fallback_note = f"gemini unavailable ({fatal[1]}); fell back to OpenAI (non-rate-limit)"
            diagnoser = Diagnoser(OpenAIProvider(openai_model(), ok), cache_namespace="openai")
            label, model = "openai (fallback)", openai_model()
            results, skipped, fatal = _run_cohorts(diagnoser, cohorts, baseline, use_cache)

    # ---- score over the cohorts that COMPLETED --------------------------------
    clear_total = clear_correct = clear_abstained = 0
    amb_total = amb_abstained = 0
    harmful = fallbacks = 0
    tok_in = tok_out = 0
    detail = []
    for co, pred in results:
        cause = pred["cause"]
        abstained = cause == INSUFFICIENT
        fallbacks += int(_is_fallback(pred))
        t = pred.get("_tokens", [0, 0]); tok_in += t[0]; tok_out += t[1]
        if co.kind == "clear":
            clear_total += 1
            if cause == co.true_cause:
                clear_correct += 1
            elif abstained:
                clear_abstained += 1
            else:
                harmful += 1
        else:
            amb_total += 1
            if abstained:
                amb_abstained += 1
            else:
                harmful += 1
        detail.append({"cohort": co.label, "kind": co.kind, "true": co.true_cause,
                       "predicted": cause, "confidence": pred.get("confidence"),
                       "n_failures": sum(co.counts.values())})

    n_completed = len(results)
    n_total = len(cohorts)
    return {
        "provider": label,
        "model": model,
        "measured_by_language_model": label.startswith(("gemini", "anthropic", "openai")),
        "partial_run": n_completed < n_total,
        "cohorts_completed": n_completed,
        "cohorts_total": n_total,
        "cohorts_skipped": skipped,
        "stop_reason": None if fatal is None else fatal[0],
        "fallback_note": fallback_note,
        "clear_cohorts": clear_total,
        "ambiguous_cohorts": amb_total,
        "accuracy_on_clear": round(clear_correct / clear_total, 3) if clear_total else None,
        "abstained_on_clear": clear_abstained,
        "abstention_rate_on_ambiguous": round(amb_abstained / amb_total, 3) if amb_total else None,
        "harmful_error_rate": round(harmful / n_completed, 3) if n_completed else None,
        "parse_failure_rate": round(fallbacks / n_completed, 3) if n_completed else None,
        "total_input_tokens": tok_in,
        "total_output_tokens": tok_out,
        "detail": detail,
    }


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    r = evaluate()
    with open(RESULTS_PATH, "w") as fh:
        json.dump(r, fh, indent=2, sort_keys=True)
    # Always also produce the offline statistical reference (all cohorts, incl. the
    # ambiguous ones that test abstention) — clearly NOT a language-model number.
    ref = evaluate(force_provider="statistical")
    with open(STATISTICAL_REF_PATH, "w") as fh:
        json.dump(ref, fh, indent=2, sort_keys=True)
    print(f"diagnosis measured by: {r['provider']}  (model: {r['model']})")
    if r["partial_run"]:
        print(f"PARTIAL RUN: {r['cohorts_completed']}/{r['cohorts_total']} cohorts "
              f"(stopped: {r['stop_reason']})")
    if r["fallback_note"]:
        print(r["fallback_note"])
    print(f"accuracy on {r['clear_cohorts']} clear cohorts:        {r['accuracy_on_clear']}")
    print(f"abstention on {r['ambiguous_cohorts']} ambiguous cohorts:  {r['abstention_rate_on_ambiguous']} (rewarded)")
    print(f"harmful-error rate:                       {r['harmful_error_rate']}")
    print(f"parse-failure rate:                       {r['parse_failure_rate']}")
    print(f"tokens: {r['total_input_tokens']} in / {r['total_output_tokens']} out")
    print(f"\n[offline statistical reference — NOT a language model — all {ref['cohorts_total']} cohorts]")
    print(f"  accuracy_clear={ref['accuracy_on_clear']}  "
          f"abstention_ambiguous={ref['abstention_rate_on_ambiguous']} (rewarded)  "
          f"harmful={ref['harmful_error_rate']}")
    print(f"\nwrote {RESULTS_PATH} and {STATISTICAL_REF_PATH}")


if __name__ == "__main__":
    main()
