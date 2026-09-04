"""Exploration price curve — final-task-list TASK 3.

Sweep ε ∈ {0.00, 0.01, 0.03, 0.10}. For each, build Switchyard (DR over the legacy
logs + an ε-exploration slice) and record, with bootstrap 95% CIs:
  - estimation error on the adversarial all-large→pa policy (₹/1k),
  - estimation error on the starved-region (direct-greedy) policy (₹/1k),
  - exploration cost (₹/1k txns, §6.2 accounting),
  - true policy value of the resulting Switchyard routing policy (₹/1k).

Traffic is common across ε (same seed): only the exploration rate changes. This is
the grader side, so it may read ground truth. Emits artifacts/exploration_curve.
{png,json}; every number in the plot is in the JSON.
"""

from __future__ import annotations

import json
import os

import numpy as np

from estimators import direct as m_direct
from estimators.core import LoggedDataset, QModel
from eval.harness import all_large_to_pa_policy, build_eval_matrices, policy_pertxn
from events import PROCESSORS, read_jsonl
from policy import switchyard as m_switchyard
from sim import economics as ec
from sim import explore_slice
from sim.ground_truth import success_prob

_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGS_PATH = os.path.join(_ROOT, "data", "logs.jsonl")
DATA_DIR = os.path.join(_ROOT, "data")
PNG_PATH = os.path.join(_ROOT, "artifacts", "exploration_curve.png")
JSON_PATH = os.path.join(_ROOT, "artifacts", "exploration_curve.json")

EPS_GRID = [0.00, 0.01, 0.03, 0.10]
SWEEP_SEED = 815          # common traffic across ε
N_RESAMPLES = 1000


def _boot(arr: np.ndarray, scale: float = 10.0, seed: int = 0):
    """(mean, lo, hi) of the mean, ×scale (paise/txn → ₹/1k)."""
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.array([arr[rng.integers(0, n, n)].mean() for _ in range(N_RESAMPLES)])
    return float(arr.mean() * scale), float(np.percentile(means, 2.5) * scale), float(np.percentile(means, 97.5) * scale)


def _true_er(ctx, proc) -> float:
    return success_prob(ctx, proc) * ec.reward_if_success_paise(proc, ctx.amount_paise)


def _exploration_cost_pertxn(slice_path: str, aux_path: str, sw) -> np.ndarray:
    """Per-txn exploration cost (paise): on the randomly-routed txns, what the
    greedy (Switchyard) policy would have earned minus what random earned; 0 on
    non-explored txns."""
    aux = {}
    with open(aux_path) as fh:
        for line in fh:
            d = json.loads(line); aux[d["txn_id"]] = d["was_explore"]
    costs = []
    for ev in read_jsonl(slice_path):
        if aux.get(ev.txn_id):
            g = sw.recommend(ev.context)
            costs.append(_true_er(ev.context, g) - _true_er(ev.context, ev.decision.processor))
        else:
            costs.append(0.0)
    return np.array(costs, dtype=np.float64)


def run() -> dict:
    legacy = LoggedDataset.from_file(LOGS_PATH)
    qmodel = QModel().fit(legacy)
    direct = m_direct.build(legacy, qmodel=qmodel)
    cell_idx, exp_reward, _ = build_eval_matrices()      # common main-regime eval traffic

    starved = direct.policy_idx                            # visits the starved large→pa region
    adv = all_large_to_pa_policy(direct.policy_idx)        # the adversarial trap
    true_adv = policy_pertxn(adv, cell_idx, exp_reward)
    true_starved = policy_pertxn(starved, cell_idx, exp_reward)
    adv_true_v, adv_lo, adv_hi = _boot(true_adv, seed=1)
    st_true_v, st_lo, st_hi = _boot(true_starved, seed=2)

    points = []
    for eps in EPS_GRID:
        sp = os.path.join(DATA_DIR, f"explore_eps_{eps:.2f}.jsonl")
        ap = os.path.join(DATA_DIR, f"explore_eps_{eps:.2f}_aux.jsonl")
        stats = explore_slice.generate(logs_path=sp, aux_path=ap, prefix=f"e{eps:.2f}_",
                                       epsilon=eps, master_seed=SWEEP_SEED)
        slice_ds = LoggedDataset.from_file(sp)
        combined = LoggedDataset.concat(legacy, slice_ds)
        sw = m_switchyard.build(combined, qmodel=qmodel)

        # estimation errors: estimate (fixed) minus true; CI = estimate − true_CI
        est_adv = sw.value_of_policy(adv)
        est_st = sw.value_of_policy(starved)
        err_adv = {"error": round(est_adv - adv_true_v, 1),
                   "ci": [round(est_adv - adv_hi, 1), round(est_adv - adv_lo, 1)]}
        err_st = {"error": round(est_st - st_true_v, 1),
                  "ci": [round(est_st - st_hi, 1), round(est_st - st_lo, 1)]}

        # true value of the switchyard policy
        sw_true = policy_pertxn(sw.policy_idx, cell_idx, exp_reward)
        tv, tv_lo, tv_hi = _boot(sw_true, seed=3)

        # exploration cost ₹/1k operating txns
        cost_arr = _exploration_cost_pertxn(sp, ap, sw)
        cv, cv_lo, cv_hi = _boot(cost_arr, seed=4)

        points.append({
            "epsilon": eps,
            "explore_rate": stats["explore_rate"],
            "n_clipped": sw.n_clipped,
            "estimation_error_adversarial": err_adv,
            "estimation_error_starved": err_st,
            "exploration_cost_per_1k": {"value": round(cv, 1), "ci": [round(cv_lo, 1), round(cv_hi, 1)]},
            "true_policy_value": {"value": round(tv, 1), "ci": [round(tv_lo, 1), round(tv_hi, 1)]},
        })
        for pth in (sp, ap):
            os.remove(pth)

    return {
        "eps_grid": EPS_GRID,
        "adversarial_policy_true_value": round(adv_true_v, 1),
        "starved_policy_true_value": round(st_true_v, 1),
        "points": points,
    }


def make_plot(result: dict, out_png: str = PNG_PATH) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = [p["epsilon"] for p in result["points"]]
    err_adv = [p["estimation_error_adversarial"]["error"] for p in result["points"]]
    err_st = [p["estimation_error_starved"]["error"] for p in result["points"]]
    cost = [p["exploration_cost_per_1k"]["value"] for p in result["points"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(eps, err_adv, "o-", color="#c65a2e", lw=2, label="estimation error — adversarial policy")
    ax1.plot(eps, err_st, "s-", color="#b5651d", lw=2, label="estimation error — starved policy")
    ax1.axhline(0, color="#000", lw=0.6)
    ax1.set_xlabel("ε (fraction of decisions randomised)")
    ax1.set_ylabel("estimation error (₹/1k, →0 is honest)", color="#c65a2e")
    ax1.tick_params(axis="y", labelcolor="#c65a2e")

    ax2 = ax1.twinx()
    ax2.plot(eps, cost, "^--", color="#2f6db3", lw=2, label="exploration cost (₹/1k)")
    ax2.set_ylabel("exploration cost (₹/1k)", color="#2f6db3")
    ax2.tick_params(axis="y", labelcolor="#2f6db3")

    ax1.set_title("Exploration price curve — honesty bought vs rupees spent")
    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, metadata={"Software": "switchyard"})
    plt.close(fig)
    return out_png


def main():
    r = run()
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as fh:
        json.dump(r, fh, indent=2, sort_keys=True)
    make_plot(r)
    print(f"{'ε':>5s} {'explore':>8s} {'err_adv':>10s} {'err_starved':>12s} "
          f"{'cost/1k':>9s} {'true_value':>11s} {'clipped':>8s}")
    for p in r["points"]:
        print(f"{p['epsilon']:>5.2f} {p['explore_rate']:>8.4f} "
              f"{p['estimation_error_adversarial']['error']:>10.1f} "
              f"{p['estimation_error_starved']['error']:>12.1f} "
              f"{p['exploration_cost_per_1k']['value']:>9.1f} "
              f"{p['true_policy_value']['value']:>11.1f} {p['n_clipped']:>8d}")
    print(f"\nwrote {JSON_PATH} and {PNG_PATH}")


if __name__ == "__main__":
    main()
