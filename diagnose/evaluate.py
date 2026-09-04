"""Score the diagnosis node — AGENT_BRIEF §8.

Two numbers, reported separately:
  - accuracy on CLEAR cohorts (did it name the right cause?),
  - abstention rate on deliberately AMBIGUOUS cohorts (did it correctly say
    INSUFFICIENT_EVIDENCE?) — abstention is rewarded, not penalised.
Plus a harmful-error rate: asserting a specific WRONG cause (on clear) or any
specific cause (on ambiguous) — the only truly bad outcome.

Writes artifacts/diagnose_results.json.
"""

from __future__ import annotations

import json
import os

from diagnose.cohorts import build_cohorts
from diagnose.node import Diagnoser, DiagnosisInput
from diagnose.schema import INSUFFICIENT

_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(_ROOT, "artifacts", "diagnose_results.json")


def evaluate(seed: int = 0, use_cache: bool = True) -> dict:
    cohorts, baseline_counts = build_cohorts(seed)
    diagnoser = Diagnoser()

    clear_total = clear_correct = clear_abstained = 0
    amb_total = amb_abstained = 0
    harmful = 0
    provider = None
    rows = []
    for co in cohorts:
        pred = diagnoser.diagnose(
            DiagnosisInput(co.label, co.window, co.counts, baseline_counts), use_cache=use_cache)
        provider = pred.get("_provider", provider)
        cause = pred["cause"]
        abstained = cause == INSUFFICIENT
        if co.kind == "clear":
            clear_total += 1
            if cause == co.true_cause:
                clear_correct += 1
            elif abstained:
                clear_abstained += 1
            else:
                harmful += 1   # asserted the wrong specific cause
        else:  # ambiguous
            amb_total += 1
            if abstained:
                amb_abstained += 1
            else:
                harmful += 1   # asserted a specific cause where none is warranted
        rows.append({"cohort": co.label, "kind": co.kind, "true": co.true_cause,
                     "predicted": cause, "confidence": pred.get("confidence"),
                     "n_failures": sum(co.counts.values())})

    return {
        "provider": provider,
        "n_cohorts": len(cohorts),
        "clear_cohorts": clear_total,
        "ambiguous_cohorts": amb_total,
        "accuracy_on_clear": round(clear_correct / clear_total, 3) if clear_total else None,
        "abstained_on_clear": clear_abstained,
        "abstention_rate_on_ambiguous": round(amb_abstained / amb_total, 3) if amb_total else None,
        "harmful_error_rate": round(harmful / len(cohorts), 3),
        "detail": rows,
    }


def main():
    r = evaluate()
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(r, fh, indent=2, sort_keys=True)
    print(f"diagnoser provider: {r['provider']}")
    print(f"accuracy on {r['clear_cohorts']} clear cohorts:       {r['accuracy_on_clear']}")
    print(f"abstention on {r['ambiguous_cohorts']} ambiguous cohorts: {r['abstention_rate_on_ambiguous']}")
    print(f"harmful-error rate (wrong assertions):    {r['harmful_error_rate']}")
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
