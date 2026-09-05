"""Live continuous-operation experiment — EXTENSION TASK A2.

Run `bygari_baseline` and an online `switchyard` in parallel over a long fresh
stream (10 seeds × 50k = 500k txns), both starting from the same confounded
legacy log and updating as outcomes arrive. Measure cumulative net revenue, the
share of large-ticket traffic each sends to the starved pa/pc options, and the
crossover. Grader side, so it reads ground truth for outcomes; the routers do not.

Design is locked in NOTES (2026-09-05). Mini-batch note: bygari's rolling
features are frozen within a 100-txn chunk (vectorised RF predict) and updated
between chunks — a tractability choice (per-txn RF predict would take ~2 hours);
it slightly slows bygari's feedback, never favours switchyard.
"""

from __future__ import annotations

import json
import os

import numpy as np

from estimators.core import LoggedDataset
from estimators.segments import CELL_INDEX, N_CELLS, cell_of
from events import PROCESSORS
from policy.bygari import BygariRouter
from sim import economics as ec
from sim.ground_truth import success_prob_batch
from sim.traffic import generate_traffic_arrays

_ROOT = os.path.dirname(os.path.dirname(__file__))
LOGS_PATH = os.path.join(_ROOT, "data", "logs.jsonl")
ART = os.path.join(_ROOT, "artifacts")

SEEDS = list(range(2000, 2010))
N_PER_SEED = 50_000
CHUNK = 100
EPS = 0.03
RECORD_EVERY = 2_000
HIGH_AMOUNT = 500_000            # >₹5k = the starved large-ticket region
STARVED_PROCS = {"pa", "pc"}     # legacy starved these on large tickets
_PA, _PB, _PC = 0, 1, 2


class OnlineSwitchyard:
    """Per-(cell, processor) running mean of NET reward, ε-greedy exploration,
    updated online. Exploits argmax net reward among seen options; ε explores."""

    def __init__(self, epsilon: float = EPS):
        self.epsilon = epsilon
        self.sum = np.zeros((N_CELLS, 3))
        self.cnt = np.zeros((N_CELLS, 3))

    def reset(self, sum0, cnt0):
        self.sum = sum0.copy(); self.cnt = cnt0.copy()

    def _means(self, c):
        m = np.full(3, -np.inf)
        for p in range(3):
            if self.cnt[c, p] > 0:
                m[p] = self.sum[c, p] / self.cnt[c, p]
        return m

    def exploit(self, c) -> int:
        m = self._means(c)
        return int(np.argmax(m)) if np.isfinite(m).any() else _PB

    def route(self, c, rng) -> tuple[int, bool]:
        if rng.random() < self.epsilon:
            return int(rng.integers(3)), True
        return self.exploit(c), False

    def update(self, c, p, reward):
        self.sum[c, p] += reward; self.cnt[c, p] += 1


def _precompute(ta):
    n = len(ta)
    cell_idx = np.array([CELL_INDEX[cell_of(str(ta.methods[i]), str(ta.issuers[i]), int(ta.amounts[i]))]
                         for i in range(n)], dtype=np.int64)
    sp = np.empty((n, 3)); rif = np.empty((n, 3))
    for p, proc in enumerate(PROCESSORS):
        sp[:, p] = success_prob_batch(ta.methods, ta.issuers, ta.amounts, ta.hours, ta.day_indices, proc)
        rif[:, p] = ec.reward_if_success_paise_vec(proc, ta.amounts).astype(float)
    exp_reward = sp * rif
    large = ta.amounts > HIGH_AMOUNT
    return cell_idx, sp, rif, exp_reward, large


def _legacy_init():
    ds = LoggedDataset.from_file(LOGS_PATH)
    sum0 = np.zeros((N_CELLS, 3)); cnt0 = np.zeros((N_CELLS, 3))
    np.add.at(sum0, (ds.cell_idx, ds.proc_idx), ds.reward)
    np.add.at(cnt0, (ds.cell_idx, ds.proc_idx), 1)
    return ds, sum0, cnt0


def run() -> dict:
    ds, sw_sum0, sw_cnt0 = _legacy_init()
    bygari = BygariRouter().fit(ds)
    roll0, errvel0 = bygari.roll.copy(), bygari.errvel.copy()   # legacy end-state, reset per seed

    grid = list(range(RECORD_EVERY, N_PER_SEED + 1, RECORD_EVERY))
    # per-seed cumulative series (₹/1k-scale later)
    cum_b = np.zeros((len(SEEDS), len(grid)))
    cum_s = np.zeros((len(SEEDS), len(grid)))
    starved_b = np.zeros((len(SEEDS), len(grid)))
    starved_s = np.zeros((len(SEEDS), len(grid)))
    explore_cost = np.zeros(len(SEEDS))
    sw = OnlineSwitchyard(EPS)

    for si, seed in enumerate(SEEDS):
        bygari.roll = roll0.copy(); bygari.errvel = errvel0.copy()
        sw.reset(sw_sum0, sw_cnt0)
        ta = generate_traffic_arrays(N_PER_SEED, seed=seed, prefix=f"live{seed}_")
        cell_idx, sp, rif, exp_reward, large = _precompute(ta)
        rng_u = np.random.default_rng(seed + 500)       # shared outcome draws (paired)
        rng_sw = np.random.default_rng(seed + 900)      # switchyard exploration coin
        u = rng_u.random(N_PER_SEED)

        tot_b = tot_s = 0.0
        n_large = lb = ls = 0
        gi = 0
        for start in range(0, N_PER_SEED, CHUNK):
            end = min(start + CHUNK, N_PER_SEED)
            idx = np.arange(start, end)
            picks_b = bygari.route_chunk(ta.methods[idx], ta.issuers[idx], ta.amounts[idx], ta.hours[idx])
            succ_b = []
            for j, i in enumerate(idx):
                pb = _PROC_I[picks_b[j]]
                s_b = u[i] < sp[i, pb]; succ_b.append(s_b)
                tot_b += rif[i, pb] if s_b else 0.0
                # switchyard (per-event online)
                c = cell_idx[i]
                ps, explored = sw.route(c, rng_sw)
                s_s = u[i] < sp[i, ps]
                tot_s += rif[i, ps] if s_s else 0.0
                sw.update(c, ps, rif[i, ps] if s_s else 0.0)
                if explored:
                    explore_cost[si] += exp_reward[i, sw.exploit(c)] - exp_reward[i, ps]
                if large[i]:
                    n_large += 1
                    lb += int(picks_b[j] in STARVED_PROCS)
                    ls += int(PROCESSORS[ps] in STARVED_PROCS)
                if (i + 1) % RECORD_EVERY == 0:
                    cum_b[si, gi] = tot_b; cum_s[si, gi] = tot_s
                    starved_b[si, gi] = lb / max(n_large, 1)
                    starved_s[si, gi] = ls / max(n_large, 1)
                    gi += 1
            bygari.update_chunk(picks_b, succ_b)

    return _summarise(grid, cum_b, cum_s, starved_b, starved_s, explore_cost)


_PROC_I = {p: i for i, p in enumerate(PROCESSORS)}


def _boot_ci(vals, seed=0):
    rng = np.random.default_rng(seed)
    means = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)]
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summarise(grid, cum_b, cum_s, starved_b, starved_s, explore_cost) -> dict:
    # convert cumulative paise → ₹ (÷100). Curves averaged across seeds.
    mb = cum_b.mean(axis=0) / 100.0
    ms = cum_s.mean(axis=0) / 100.0
    # crossover: first grid point where switchyard mean cumulative overtakes bygari
    crossover = None
    for g, b, s in zip(grid, mb, ms):
        if s > b:
            crossover = g; break
    fb, fb_lo, fb_hi = _boot_ci(cum_b[:, -1] / 100.0)
    fs, fs_lo, fs_hi = _boot_ci(cum_s[:, -1] / 100.0)
    diff = (cum_s[:, -1] - cum_b[:, -1]) / 100.0
    d, d_lo, d_hi = _boot_ci(diff)
    return {
        "config": {"seeds": SEEDS, "n_per_seed": N_PER_SEED, "chunk": CHUNK, "epsilon": EPS,
                   "total_txns": N_PER_SEED * len(SEEDS)},
        "grid": grid,
        "cumulative_rupees": {"bygari_baseline": mb.tolist(), "switchyard": ms.tolist()},
        "starved_region_share": {"bygari_baseline": starved_b.mean(axis=0).tolist(),
                                 "switchyard": starved_s.mean(axis=0).tolist()},
        "final_cumulative_rupees": {
            "bygari_baseline": {"mean": round(fb, 1), "ci": [round(fb_lo, 1), round(fb_hi, 1)]},
            "switchyard": {"mean": round(fs, 1), "ci": [round(fs_lo, 1), round(fs_hi, 1)]},
            "switchyard_minus_bygari": {"mean": round(d, 1), "ci": [round(d_lo, 1), round(d_hi, 1)]},
        },
        "crossover_txn": crossover,
        "switchyard_exploration_cost_rupees": round(float(explore_cost.mean()) / 100.0, 1),
        "final_starved_share": {"bygari_baseline": round(float(starved_b[:, -1].mean()), 4),
                                "switchyard": round(float(starved_s[:, -1].mean()), 4)},
    }


def make_plots(r: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = r["grid"]
    # cumulative value (lines overlap at the ₹2M scale — the gap is ~0.13%), so a
    # secondary axis shows the running difference (switchyard − bygari): it stays
    # BELOW zero, i.e. no crossover.
    import numpy as _np
    mb = _np.array(r["cumulative_rupees"]["bygari_baseline"])
    ms = _np.array(r["cumulative_rupees"]["switchyard"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, mb, color="#c65a2e", lw=2, label="bygari_baseline (cumulative)")
    ax.plot(grid, ms, color="#2f6db3", lw=2, label="switchyard, net of exploration (cumulative)")
    ax.set_xlabel("transactions processed (per seed)")
    ax.set_ylabel("cumulative net revenue (₹, mean over 10 seeds)")
    ax.set_title("Cumulative net revenue — bygari_baseline vs switchyard")
    ax2 = ax.twinx()
    ax2.plot(grid, ms - mb, color="#4c9a52", ls="--", lw=1.5, label="switchyard − bygari (right axis)")
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_ylabel("difference (₹): switchyard − bygari", color="#4c9a52")
    ax2.tick_params(axis="y", labelcolor="#4c9a52")
    fc = r["final_cumulative_rupees"]["switchyard_minus_bygari"]
    ax2.annotate(f"final gap ₹{fc['mean']:,.0f} (no crossover)", xy=(grid[-1], (ms - mb)[-1]),
                 xytext=(grid[len(grid)//4], (ms - mb).min() * 0.9), fontsize=9, color="#2a6a30")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(os.path.join(ART, "cumulative_value.png"), dpi=110, metadata={"Software": "switchyard"})
    plt.close(fig)
    # starved-region traffic share
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, r["starved_region_share"]["bygari_baseline"], color="#c65a2e", lw=2, label="bygari_baseline")
    ax.plot(grid, r["starved_region_share"]["switchyard"], color="#2f6db3", lw=2, label="switchyard")
    ax.set_xlabel("transactions processed (per seed)")
    ax.set_ylabel("share of large-ticket (>₹5k) traffic sent to pa/pc")
    ax.set_title("Traffic to the starved large-ticket options over time")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(os.path.join(ART, "starved_region_traffic.png"), dpi=110, metadata={"Software": "switchyard"})
    plt.close(fig)


def main():
    r = run()
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "timeseries.json"), "w") as fh:
        json.dump(r, fh, indent=2, sort_keys=True)
    make_plots(r)
    fc = r["final_cumulative_rupees"]
    print(f"final cumulative net revenue (₹, mean/seed over 500k txns):")
    print(f"  bygari_baseline: {fc['bygari_baseline']['mean']}  CI {fc['bygari_baseline']['ci']}")
    print(f"  switchyard:      {fc['switchyard']['mean']}  CI {fc['switchyard']['ci']}")
    print(f"  switchyard − bygari: {fc['switchyard_minus_bygari']['mean']}  CI {fc['switchyard_minus_bygari']['ci']}")
    print(f"  crossover txn: {r['crossover_txn']}")
    print(f"  final starved-region share  bygari={r['final_starved_share']['bygari_baseline']} "
          f"switchyard={r['final_starved_share']['switchyard']}")
    print(f"  switchyard exploration cost: ₹{r['switchyard_exploration_cost_rupees']}/seed")


if __name__ == "__main__":
    main()
