"""Three-way, two-world diagnosis comparison — TASK C2 + C3.

Runs three methods — a hand-written rule, a small trained classifier, and an LLM
— on both the closed-world cohorts (documented codes) and the held-out open-world
cohorts (withheld real codes, free-text, red herrings). Reports, per method-world:
accuracy on clear cohorts, harmful-error rate, and (C3) abstention split into the
deterministic sample-size guardrail vs the method's OWN judgment.

The LLM provider is resolved exactly like diagnose/evaluate.py (committed OpenAI
cache reproduces with no key). Rule and trained classifier are deterministic and
need no key. Writes artifacts/threeway_results.json.
"""

from __future__ import annotations

import json
import os

from diagnose.cohorts import build_cohorts
from diagnose.evaluate import _make_primary
from diagnose.node import CacheMiss, Diagnoser, DiagnosisInput, StatisticalProvider
from diagnose.openworld import build_open_world_cohorts
from diagnose.schema import INSUFFICIENT
from diagnose.trained import TrainedClassifier

_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(_ROOT, "artifacts", "threeway_results.json")


def _counts(cohort) -> dict:
    return getattr(cohort, "cohort_counts", None) or getattr(cohort, "counts", {})


def _to_input(cohort, baseline) -> DiagnosisInput:
    return DiagnosisInput(cohort.label, cohort.window, _counts(cohort), baseline,
                          unknown_codes=dict(getattr(cohort, "unknown_codes", {}) or {}),
                          messages=tuple(getattr(cohort, "messages", ()) or ()))


def _run(diagnoser, cohorts, baseline, use_cache=True):
    out = []
    for co in cohorts:
        try:
            pred = diagnoser.diagnose(_to_input(co, baseline), use_cache=use_cache)
        except CacheMiss:
            pred = {"cause": INSUFFICIENT, "confidence": 0.0, "_abstain_kind": "cache_miss",
                    "evidence": ["cache miss (no key); skipped"]}
        out.append((co, pred))
    return out


def _score(results: list) -> dict:
    clear_total = clear_correct = 0
    amb_total = amb_abstained = 0
    harmful = 0
    guardrail_abst = model_abst = 0
    reached_model = reached_model_abst = 0   # cohorts that passed the guardrail
    detail = []
    for co, pred in results:
        cause = pred["cause"]
        abstained = cause == INSUFFICIENT
        kind = pred.get("_abstain_kind")
        passed_guardrail = kind != "guardrail"
        if passed_guardrail:
            reached_model += 1
        if abstained:
            if kind == "guardrail":
                guardrail_abst += 1
            else:
                model_abst += 1
                reached_model_abst += 1
        if co.kind == "clear":
            clear_total += 1
            if cause == co.true_cause:
                clear_correct += 1
            elif not abstained:
                harmful += 1
        else:
            amb_total += 1
            if abstained:
                amb_abstained += 1
            else:
                harmful += 1
        detail.append({"cohort": co.label, "kind": co.kind, "true": co.true_cause,
                       "predicted": cause, "abstain_kind": kind})
    n = len(results)
    return {
        "n_cohorts": n,
        "clear_cohorts": clear_total,
        "ambiguous_cohorts": amb_total,
        "accuracy_on_clear": round(clear_correct / clear_total, 3) if clear_total else None,
        "harmful_error_rate": round(harmful / n, 3) if n else None,
        "abstention_on_ambiguous_incl_guardrail": round(amb_abstained / amb_total, 3) if amb_total else None,
        # C3: the method's OWN abstention rate, on cohorts that passed the guardrail
        "model_own_abstention_rate": round(reached_model_abst / reached_model, 3) if reached_model else None,
        "guardrail_abstentions": guardrail_abst,
        "model_abstentions": model_abst,
        "detail": detail,
    }


def evaluate(use_cache: bool = True, force_provider: str | None = None) -> dict:
    from diagnose.node import load_env
    load_env()
    closed, baseline = build_cohorts(0)
    openw = build_open_world_cohorts()

    rule = Diagnoser(StatisticalProvider(), cache_namespace="statistical")
    trained_clf = TrainedClassifier().fit()
    trained = Diagnoser(trained_clf, cache_namespace="trained")
    llm, llm_label, llm_model = _make_primary(force_provider)

    methods = [("rule", rule), ("trained", trained), ("llm", llm)]
    out = {"llm_provider": llm_label, "llm_model": llm_model, "closed": {}, "open": {}}
    for world, cohorts in (("closed", closed), ("open", openw)):
        for name, dg in methods:
            out[world][name] = _score(_run(dg, cohorts, baseline, use_cache))
    return out


def _cell(s: dict) -> str:
    return (f"acc_clear={s['accuracy_on_clear']} harmful={s['harmful_error_rate']} "
            f"abst(all/model)={s['abstention_on_ambiguous_incl_guardrail']}/{s['model_own_abstention_rate']}")


def main():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    r = evaluate()
    with open(RESULTS_PATH, "w") as fh:
        json.dump(r, fh, indent=2, sort_keys=True)
    print(f"LLM: {r['llm_provider']} ({r['llm_model']})\n")
    print(f"{'method':10s} | {'CLOSED-WORLD':45s} | OPEN-WORLD")
    print("-" * 110)
    for m in ("rule", "trained", "llm"):
        print(f"{m:10s} | {_cell(r['closed'][m]):45s} | {_cell(r['open'][m])}")
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
