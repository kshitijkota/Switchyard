"""Held-out regime — AGENT_BRIEF §6.1 / §13 / ABSOLUTE RULE 5.

A DIFFERENT traffic mix (more card, uniform issuers, fatter tail, fewer days) and
a DIFFERENT degradation schedule (day%7 ∈ {1,4} instead of {2,5}). The four
methods are NOT retrained — they keep the policies and self-estimates learned on
the main regime; we only re-measure TRUE value here. Run exactly once.

This is a robustness check: does chowk's honesty advantage survive a regime shift?
"""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np

from estimators.build import build_all, load_datasets
from eval.harness import (
    build_eval_matrices, evaluate_policy_honesty, policy_pertxn, true_best_pertxn,
    all_large_to_pa_policy,
)

_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(_ROOT, "artifacts", "heldout_results.json")

HELDOUT_SEEDS = list(range(9000, 9010))
DEGRADE_RESIDUES = (1, 4)   # different schedule from the main regime's (2, 5)


def run_heldout() -> dict:
    methods = build_all(*load_datasets())     # trained on the MAIN logs only
    cell_idx, exp_reward, legacy_pertxn = build_eval_matrices(
        HELDOUT_SEEDS, profile="heldout", degrade_residues=DEGRADE_RESIDUES)

    legacy_true = float(legacy_pertxn.mean() * 10.0)
    oracle = float(true_best_pertxn(exp_reward).mean() * 10.0)

    table = []
    for name, m in methods.items():
        est = m.estimated_value_per_1k()   # self-estimate from the MAIN regime
        true = float(policy_pertxn(m.policy_idx, cell_idx, exp_reward).mean() * 10.0)
        table.append({"method": name, "estimated_value_main": round(est, 1),
                      "true_value_heldout": round(true, 1),
                      "estimation_error": round(est - true, 1)})

    return {
        "run_timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "regime": {"traffic_profile": "heldout", "degrade_residues": list(DEGRADE_RESIDUES),
                   "seeds": HELDOUT_SEEDS, "n_eval_txns": len(cell_idx)},
        "legacy_baseline_value": round(legacy_true, 1),
        "oracle_value": round(oracle, 1),
        "results_table": table,
        "ope_honesty": evaluate_policy_honesty(
            methods, methods["direct"].policy_idx,
            "direct-greedy on held-out traffic", cell_idx, exp_reward),
        "adversarial_honesty": evaluate_policy_honesty(
            methods, all_large_to_pa_policy(methods["direct"].policy_idx),
            "adversarial all-large→pa on held-out traffic", cell_idx, exp_reward),
    }


def main():
    if os.path.exists(RESULTS_PATH):
        print(f"REFUSING to re-run: {RESULTS_PATH} already exists (ABSOLUTE RULE 5 — "
              f"never open the held-out regime more than once). Delete it only if you "
              f"intend to intentionally re-open the held-out set.")
        return
    r = run_heldout()
    with open(RESULTS_PATH, "w") as fh:
        json.dump(r, fh, indent=2, sort_keys=True)
    print(f"HELD-OUT REGIME — single evaluation at {r['run_timestamp']}")
    print(f"legacy {r['legacy_baseline_value']}  oracle {r['oracle_value']} ₹/1k\n")
    print(f"{'method':7s} {'est(main)':>10s} {'true(heldout)':>14s} {'error':>8s}")
    for row in r["results_table"]:
        print(f"{row['method']:7s} {row['estimated_value_main']:>10.1f} "
              f"{row['true_value_heldout']:>14.1f} {row['estimation_error']:>8.1f}")
    o = r["ope_honesty"]
    print(f"\nOPE honesty ({o['target_policy']}) true {o['true_value']}:")
    for name, e in o["estimators"].items():
        print(f"  {name:7s} error {e['error']:+9.1f}")
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
