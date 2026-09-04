"""Evaluation harness — AGENT_BRIEF §6.1 / §6.2.

Because the simulator is under our control we compute TRUE policy value by rolling
each learned policy forward through fresh simulated traffic (common random numbers
across methods, 10 seeds). This module is the grader side, so it MAY read ground
truth; the methods it grades never do.

Produces:
  - the results table (estimated vs true value, estimation error, 95% CI),
  - the off-policy-evaluation honesty benchmark (each estimator's estimate of a
    fixed starved-cell-visiting policy vs the truth),
  - the hdfc×upi segment decision table vs the true best,
  - the §6.2 exploration accounting,
and writes them to artifacts/results.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

from estimators.build import build_all, load_datasets
from estimators.segments import (
    ALL_CELLS, BUCKET_LABELS, CELL_INDEX, Cell, N_CELLS, PROC_INDEX, cell_of,
    representative_context,
)
from eval.bootstrap import paired_bootstrap
from events import PROCESSORS
from policy.legacy import probs as legacy_probs
from sim import economics as ec
from sim.ground_truth import success_prob
from sim.traffic import generate_traffic_arrays

_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(_ROOT, "artifacts", "results.json")

EVAL_SEEDS = list(range(2000, 2010))   # 10 seeds, common across methods
EVAL_N_PER_SEED = 20_000


def build_eval_matrices(seeds=EVAL_SEEDS, n_per_seed=EVAL_N_PER_SEED):
    """Return (cell_idx, exp_reward, legacy_pertxn) over the pooled eval traffic.

    exp_reward[i, p] = true expected net reward (paise) of routing txn i to
    processor p = success_prob(x_i, p) * reward_if_success(p, amount_i).
    """
    ms, iss, amt, hr, day = [], [], [], [], []
    for s in seeds:
        ta = generate_traffic_arrays(n_per_seed, seed=s, prefix=f"ev{s}_")
        ms.append(ta.methods); iss.append(ta.issuers); amt.append(ta.amounts)
        hr.append(ta.hours); day.append(ta.day_indices)
    methods = np.concatenate(ms); issuers = np.concatenate(iss)
    amounts = np.concatenate(amt); hours = np.concatenate(hr); days = np.concatenate(day)
    n = len(amounts)

    from sim.ground_truth import success_prob_batch
    exp_reward = np.empty((n, len(PROCESSORS)), dtype=np.float64)
    for p, proc in enumerate(PROCESSORS):
        sp = success_prob_batch(methods, issuers, amounts, hours, days, proc)
        exp_reward[:, p] = sp * ec.reward_if_success_paise_vec(proc, amounts).astype(np.float64)

    # cell index per eval txn
    cell_idx = np.array([CELL_INDEX[cell_of(str(methods[i]), str(issuers[i]), int(amounts[i]))]
                         for i in range(n)], dtype=np.int64)

    # legacy stochastic baseline expected reward per txn
    high = amounts > 500_000
    upi = methods == "upi"
    pa_p = np.where(high, 0.03, np.where(upi, 0.90, 0.10))
    pb_p = np.where(high, 0.95, np.where(upi, 0.03, 0.05))
    pc_p = np.where(high, 0.02, np.where(upi, 0.07, 0.85))
    legacy_pertxn = pa_p * exp_reward[:, 0] + pb_p * exp_reward[:, 1] + pc_p * exp_reward[:, 2]
    return cell_idx, exp_reward, legacy_pertxn


def policy_pertxn(policy_idx: np.ndarray, cell_idx: np.ndarray, exp_reward: np.ndarray) -> np.ndarray:
    chosen = policy_idx[cell_idx]
    return exp_reward[np.arange(len(cell_idx)), chosen]


def true_best_pertxn(exp_reward: np.ndarray) -> np.ndarray:
    return exp_reward.max(axis=1)


# --- Segment decision table -----------------------------------------------------

def segment_table(methods: dict, cell_idx, exp_reward) -> list[dict]:
    """hdfc×upi picks vs the true best. 'True best' is the argmax of the cell's
    AVERAGE true expected reward over the eval traffic — matching what the
    per-cell methods actually optimise (a single representative amount would be an
    unfair yardstick for a bucket that spans a fee crossover)."""
    rows = []
    for b in range(len(BUCKET_LABELS)):
        cell = Cell("upi", "hdfc", b)
        ci = CELL_INDEX[cell]
        in_cell = cell_idx == ci
        if in_cell.any():
            avg = exp_reward[in_cell].mean(axis=0)
            true_best = PROCESSORS[int(avg.argmax())]
            true_vals = {p: round(float(avg[k])) for k, p in enumerate(PROCESSORS)}
        else:
            true_best, true_vals = "?", {}
        rows.append({
            "segment": cell.label(), "bucket": BUCKET_LABELS[b],
            "true_best": true_best, "true_expected_reward": true_vals,
            "picks": {name: m.recommend_cell(cell) for name, m in methods.items()},
        })
    return rows


def pa_extrapolation_evidence(methods: dict, out_json: str | None = None) -> dict:
    """The unambiguous 'cannot learn what you never tried' evidence: direct's
    Q-model predicted pa success vs the TRUE pa success across ticket sizes, on a
    fixed segment. Above ₹5k the legacy policy never sent traffic to pa, so the
    model keeps extrapolating pa's healthy small-ticket rate while the truth has
    dropped 0.18."""
    q = methods["direct"]._qmodel
    amounts = np.array([20_000, 60_000, 200_000, 400_000, 700_000, 1_200_000, 3_000_000, 8_000_000])
    ms = np.array(["card"] * len(amounts)); iss = np.array(["hdfc"] * len(amounts))
    hrs = np.full(len(amounts), 12)
    pred_pa = q.success_proba(ms, iss, amounts, hrs)[:, PROC_INDEX["pa"]]
    from events import Context
    from datetime import datetime
    true_pa = np.array([
        success_prob(Context(f"x{i}", datetime(2026, 1, 1, 12), "card", "hdfc", int(a), 12), "pa")
        for i, a in enumerate(amounts)])
    rows = [{"amount_rupees": int(a // 100), "predicted_pa_success": round(float(pp), 3),
             "true_pa_success": round(float(tp), 3), "gap": round(float(pp - tp), 3)}
            for a, pp, tp in zip(amounts, pred_pa, true_pa)]
    out = {"segment": "card × hdfc", "note": "legacy sends >₹5,000 to pb, starving pa; "
           "direct extrapolates pa's small-ticket success and never learns the −0.18 drop.",
           "rows": rows}
    if out_json:
        with open(out_json, "w") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    return out


# --- Off-policy-evaluation honesty benchmark ------------------------------------

def evaluate_policy_honesty(methods: dict, target: np.ndarray, label: str,
                            cell_idx, exp_reward) -> dict:
    """Point every estimator at a FIXED policy and ask each what it is worth. The
    truth is known by rollout; the honest estimator wins."""
    true_value = float(policy_pertxn(target, cell_idx, exp_reward).mean() * 10.0)
    rows = {}
    for name, m in methods.items():
        est = m.value_of_policy(target)
        rows[name] = {"estimate": round(est, 1), "error": round(est - true_value, 1),
                      "coverage": round(m.policy_coverage(target), 4)}
    return {"target_policy": label, "true_value": round(true_value, 1), "estimators": rows}


def all_large_to_pa_policy(base_policy: np.ndarray) -> np.ndarray:
    """An adversarial policy: take the direct policy but dump ALL large tickets on
    pa — exactly the trap direct's own model walks into. direct will value this
    highly (it thinks pa is great on large); chowk, having explored, will not."""
    pol = base_policy.copy()
    pa = PROC_INDEX["pa"]
    for c, cell in enumerate(ALL_CELLS):
        if cell.bucket in (3, 4):
            pol[c] = pa
    return pol


def naive_success_policy(qmodel) -> np.ndarray:
    """§1's fee-blind strawman: route to the highest predicted SUCCESS rate,
    ignoring fees. Loses money at small tickets where the flat fee bites."""
    ms = np.array([c.method for c in ALL_CELLS])
    iss = np.array([c.issuer for c in ALL_CELLS])
    amt = np.array([representative_context(c).amount_paise for c in ALL_CELLS])
    hr = np.full(N_CELLS, 12)
    sp = qmodel.success_proba(ms, iss, amt, hr)
    return sp.argmax(axis=1).astype(np.int64)


# --- Exploration accounting (§6.2) ----------------------------------------------

def exploration_accounting(methods: dict, explore_path: str, aux_path: str) -> dict:
    """Rupee cost of the 3% randomly-routed slice: what the greedy (chowk) policy
    would have earned on those txns minus what random routing actually earned
    (expected reward, ground truth)."""
    from events import read_jsonl
    chowk = methods["chowk"]
    aux = {}
    with open(aux_path) as fh:
        for line in fh:
            d = json.loads(line)
            aux[d["txn_id"]] = d
    greedy_minus_random = 0.0
    n_explore = 0
    total_reward_defensible = 0.0
    n_total = 0
    for ev in read_jsonl(explore_path):
        ctx = ev.context
        n_total += 1
        greedy_proc = chowk.recommend(ctx)
        total_reward_defensible += success_prob(ctx, greedy_proc) * ec.reward_if_success_paise(greedy_proc, ctx.amount_paise)
        if aux[ctx.txn_id]["was_explore"]:
            n_explore += 1
            chosen = ev.decision.processor
            greedy_r = success_prob(ctx, greedy_proc) * ec.reward_if_success_paise(greedy_proc, ctx.amount_paise)
            random_r = success_prob(ctx, chosen) * ec.reward_if_success_paise(chosen, ctx.amount_paise)
            greedy_minus_random += (greedy_r - random_r)
    spent_rupees = greedy_minus_random / 100.0
    defensible_rupees = total_reward_defensible / 100.0
    return {
        "n_explore_txns": n_explore,
        "n_total_txns": n_total,
        "spent_rupees": round(spent_rupees, 2),
        "defensible_routed_value_rupees": round(defensible_rupees, 2),
        "sentence": f"Spent ₹{spent_rupees:,.0f} to keep ₹{defensible_rupees:,.0f} "
                    f"of routing decisions defensible.",
    }


# --- Orchestration --------------------------------------------------------------

def run_evaluation(seeds=EVAL_SEEDS, n_per_seed=EVAL_N_PER_SEED) -> dict:
    legacy_ds, explore_ds = load_datasets()
    methods = build_all(legacy_ds, explore_ds)
    cell_idx, exp_reward, legacy_pertxn = build_eval_matrices(seeds, n_per_seed)

    per_txn = {name: policy_pertxn(m.policy_idx, cell_idx, exp_reward) for name, m in methods.items()}
    boot = paired_bootstrap(per_txn, legacy_pertxn)

    # optimal (oracle) value + §1 fee-blind success-router baseline, for reference
    oracle = float(true_best_pertxn(exp_reward).mean() * 10.0)
    naive_pol = naive_success_policy(methods["direct"]._qmodel)
    naive_value = float(policy_pertxn(naive_pol, cell_idx, exp_reward).mean() * 10.0)

    table = []
    for name, m in methods.items():
        est = m.estimated_value_per_1k()
        b = boot[name]
        table.append({
            "method": name,
            "estimated_value": round(est, 1),
            "true_value": round(b["true_value"], 1),
            "estimation_error": round(est - b["true_value"], 1),
            "true_value_ci": [round(b["value_ci"][0], 1), round(b["value_ci"][1], 1)],
            "improvement_over_legacy": round(b["improvement"], 1),
            "improvement_ci": [round(b["improvement_ci"][0], 1), round(b["improvement_ci"][1], 1)],
            "distinguishable_from_baseline": b["distinguishable_from_baseline"],
            "n_clipped": m.n_clipped,
        })

    results = {
        "config": {"eval_seeds": seeds, "n_per_seed": n_per_seed,
                   "n_eval_txns": len(cell_idx), "bootstrap_resamples": 1000},
        "legacy_baseline_value": round(boot["_baseline"]["true_value"], 1),
        "naive_success_router_value": round(naive_value, 1),
        "oracle_value": round(oracle, 1),
        "results_table": table,
        "ope_honesty": evaluate_policy_honesty(
            methods, methods["direct"].policy_idx,
            "direct-greedy (visits starved large→pa)", cell_idx, exp_reward),
        "adversarial_honesty": evaluate_policy_honesty(
            methods, all_large_to_pa_policy(methods["direct"].policy_idx),
            "adversarial: ALL large tickets → pa (the trap)", cell_idx, exp_reward),
        "segment_table": segment_table(methods, cell_idx, exp_reward),
        "pa_extrapolation": pa_extrapolation_evidence(
            methods, os.path.join(_ROOT, "artifacts", "extrapolation.json")),
        "exploration_accounting": exploration_accounting(
            methods, os.path.join(_ROOT, "data", "explore.jsonl"),
            os.path.join(_ROOT, "data", "explore_aux.jsonl")),
    }
    return results


def print_results(r: dict) -> None:
    print(f"Legacy baseline true value: {r['legacy_baseline_value']} ₹/1k   "
          f"(oracle {r['oracle_value']} ₹/1k)\n")
    print(f"{'method':7s} {'estimated':>10s} {'true':>10s} {'error':>8s} "
          f"{'true 95% CI':>20s} {'improv (CI)':>22s} clip")
    for row in r["results_table"]:
        ci = f"[{row['true_value_ci'][0]:.0f},{row['true_value_ci'][1]:.0f}]"
        imp = (f"{row['improvement_over_legacy']:.0f} "
               f"[{row['improvement_ci'][0]:.0f},{row['improvement_ci'][1]:.0f}]")
        note = "" if row["distinguishable_from_baseline"] else "  cannot distinguish"
        print(f"{row['method']:7s} {row['estimated_value']:10.1f} {row['true_value']:10.1f} "
              f"{row['estimation_error']:8.1f} {ci:>20s} {imp:>22s} {row['n_clipped']}{note}")

    print(f"\nNaive success-rate router (fee-blind, §1): {r['naive_success_router_value']} ₹/1k")

    for key in ("ope_honesty", "adversarial_honesty"):
        o = r[key]
        print(f"\nOPE honesty — {o['target_policy']}   (true value {o['true_value']:.1f} ₹/1k)")
        for name, e in o["estimators"].items():
            print(f"  {name:7s} estimate {e['estimate']:9.1f}  error {e['error']:+9.1f}  "
                  f"coverage {e['coverage']*100:5.1f}%")

    print("\nSegment decision table — hdfc × upi (true best vs picks):")
    for row in r["segment_table"]:
        picks = " ".join(f"{k}={'✓' if v==row['true_best'] else v+'✗'}" for k, v in row["picks"].items())
        print(f"  {row['bucket']:9s} true={row['true_best']}   {picks}")

    print("\ndirect's pa success: predicted vs true by ticket size "
          "(the extrapolation failure):")
    print(f"  {'₹amount':>9s} {'predicted':>10s} {'true':>7s} {'gap':>7s}")
    for row in r["pa_extrapolation"]["rows"]:
        print(f"  {row['amount_rupees']:>9,d} {row['predicted_pa_success']:>10.3f} "
              f"{row['true_pa_success']:>7.3f} {row['gap']:>+7.3f}")

    print("\n" + r["exploration_accounting"]["sentence"])


def main():
    r = run_evaluation()
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(r, fh, indent=2, sort_keys=True)
    print_results(r)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
