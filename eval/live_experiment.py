"""Live continuous-operation experiment — EXTENSION TASK A2 + diagnostic re-run.

`bygari_baseline` vs an online `switchyard` bandit over a fresh stream, both
starting from the same confounded legacy log and updating online. Grader side
(reads ground truth for outcomes; the routers do not).

Diagnostic re-run (design locked in NOTES 2026-09-05):
  - precision: 40 seeds (2000–2039), 50k/seed, common random numbers;
  - H1: ε ∈ {0.01, 0.03, 0.10};
  - H2: best ε extended to 2,000,000 txns/seed (10 seeds);
  - H3: net difference restricted to the starved large-ticket region.
Seeds run in parallel processes; RF n_jobs=1 in workers. Mini-batch note: bygari
rolling features frozen within a 100-txn chunk (tractability; slightly slows
bygari feedback, never favours switchyard).
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool

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

SEEDS = list(range(2000, 2040))          # 40 seeds (frozen in NOTES)
N_PER_SEED = 50_000
EPS_GRID = [0.01, 0.03, 0.10]
CHUNK = 100
HIGH_AMOUNT = 500_000                     # >₹5k = starved large-ticket region
STARVED_PROCS = {"pa", "pc"}
_PROC_I = {p: i for i, p in enumerate(PROCESSORS)}
N_WORKERS = 10


class OnlineSwitchyard:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.sum = np.zeros((N_CELLS, 3)); self.cnt = np.zeros((N_CELLS, 3))

    def reset(self, sum0, cnt0):
        self.sum = sum0.copy(); self.cnt = cnt0.copy()

    def exploit(self, c):
        m = np.where(self.cnt[c] > 0, self.sum[c] / np.where(self.cnt[c] > 0, self.cnt[c], 1), -np.inf)
        return int(np.argmax(m)) if np.isfinite(m).any() else _PROC_I["pb"]

    def route(self, c, rng):
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
    return cell_idx, sp, rif, sp * rif, (ta.amounts > HIGH_AMOUNT)


def _legacy_init():
    ds = LoggedDataset.from_file(LOGS_PATH)
    sum0 = np.zeros((N_CELLS, 3)); cnt0 = np.zeros((N_CELLS, 3))
    np.add.at(sum0, (ds.cell_idx, ds.proc_idx), ds.reward)
    np.add.at(cnt0, (ds.cell_idx, ds.proc_idx), 1)
    return ds, sum0, cnt0


# --- worker (parallel over seeds) ----------------------------------------------
_W = {}


def _init_worker(bygari, sum0, cnt0):
    bygari.rf.n_jobs = 1
    _W["bygari"] = bygari
    _W["roll0"] = bygari.roll.copy(); _W["errvel0"] = bygari.errvel.copy()
    _W["sum0"] = sum0; _W["cnt0"] = cnt0


def _run_seed(seed, epsilon, n_per_seed):
    bygari = _W["bygari"]
    bygari.roll = _W["roll0"].copy(); bygari.errvel = _W["errvel0"].copy()
    sw = OnlineSwitchyard(epsilon); sw.reset(_W["sum0"], _W["cnt0"])
    ta = generate_traffic_arrays(n_per_seed, seed=seed, prefix=f"live{seed}_")
    cell_idx, sp, rif, exp_reward, large = _precompute(ta)
    u = np.random.default_rng(seed + 500).random(n_per_seed)
    rng_sw = np.random.default_rng(seed + 900)

    record_every = max(1, n_per_seed // 25)
    grid, cb, cs, sb, ss = [], [], [], [], []
    tot_b = tot_s = 0.0
    large_b = large_s = small_b = small_s = 0.0
    n_large = lb = ls = 0
    explore_cost = 0.0

    for start in range(0, n_per_seed, CHUNK):
        end = min(start + CHUNK, n_per_seed)
        idx = np.arange(start, end)
        picks_b = bygari.route_chunk(ta.methods[idx], ta.issuers[idx], ta.amounts[idx], ta.hours[idx])
        succ_b = []
        for j, i in enumerate(idx):
            pb = _PROC_I[picks_b[j]]
            s_b = u[i] < sp[i, pb]; succ_b.append(s_b)
            r_b = rif[i, pb] if s_b else 0.0; tot_b += r_b
            c = cell_idx[i]
            ps, explored = sw.route(c, rng_sw)
            s_s = u[i] < sp[i, ps]
            r_s = rif[i, ps] if s_s else 0.0; tot_s += r_s
            sw.update(c, ps, r_s)
            if explored:
                explore_cost += exp_reward[i, sw.exploit(c)] - exp_reward[i, ps]
            if large[i]:
                n_large += 1; large_b += r_b; large_s += r_s
                lb += int(picks_b[j] in STARVED_PROCS); ls += int(PROCESSORS[ps] in STARVED_PROCS)
            else:
                small_b += r_b; small_s += r_s
            if (i + 1) % record_every == 0:
                grid.append(int(i + 1)); cb.append(tot_b); cs.append(tot_s)
                sb.append(lb / max(n_large, 1)); ss.append(ls / max(n_large, 1))
        bygari.update_chunk(picks_b, succ_b)

    return {"seed": seed, "grid": grid, "cum_b": cb, "cum_s": cs, "starved_b": sb, "starved_s": ss,
            "final_b": tot_b, "final_s": tot_s, "n_large": n_large, "n_total": n_per_seed,
            "large_b": large_b, "large_s": large_s, "small_b": small_b, "small_s": small_s,
            "explore_cost": explore_cost}


def _np_default(o):
    """Belt-and-suspenders JSON encoder for stray numpy scalars/arrays."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _save(obj):
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "timeseries.json"), "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_np_default)


def _boot(vals, seed=0):
    vals = np.asarray(vals, float); rng = np.random.default_rng(seed)
    m = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(1000)]
    return float(vals.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_cell(epsilon, seeds, n_per_seed) -> dict:
    ds, sum0, cnt0 = _legacy_init()
    bygari = BygariRouter().fit(ds)
    with Pool(N_WORKERS, initializer=_init_worker, initargs=(bygari, sum0, cnt0)) as pool:
        res = pool.starmap(_run_seed, [(s, epsilon, n_per_seed) for s in seeds])
    res.sort(key=lambda r: r["seed"])
    grid = res[0]["grid"]
    cum_b = np.array([r["cum_b"] for r in res]); cum_s = np.array([r["cum_s"] for r in res])
    starved_b = np.array([r["starved_b"] for r in res]); starved_s = np.array([r["starved_s"] for r in res])
    fb = np.array([r["final_b"] for r in res]) / 100.0
    fs = np.array([r["final_s"] for r in res]) / 100.0
    mb = cum_b.mean(axis=0) / 100.0; ms = cum_s.mean(axis=0) / 100.0
    crossover = next((int(g) for g, b, s in zip(grid, mb, ms) if s > b), None)
    d, d_lo, d_hi = _boot(fs - fb, seed=1)
    # H3 — restricted to the starved large-ticket region and remainder
    lb = np.array([r["large_b"] for r in res]) / 100.0; ls = np.array([r["large_s"] for r in res]) / 100.0
    sb = np.array([r["small_b"] for r in res]) / 100.0; ss = np.array([r["small_s"] for r in res]) / 100.0
    ld, ld_lo, ld_hi = _boot(ls - lb, seed=2)
    sd, sd_lo, sd_hi = _boot(ss - sb, seed=3)
    starved_share = float(np.mean([r["n_large"] / r["n_total"] for r in res]))
    return {
        "epsilon": epsilon, "seeds": seeds, "n_per_seed": n_per_seed, "total_txns": n_per_seed * len(seeds),
        "grid": grid, "cum_b_mean": mb.tolist(), "cum_s_mean": ms.tolist(),
        "starved_b_mean": starved_b.mean(axis=0).tolist(), "starved_s_mean": starved_s.mean(axis=0).tolist(),
        "final": {"bygari": [round(float(fb.mean()), 1)] + [round(x, 1) for x in _boot(fb)[1:]],
                  "switchyard": [round(float(fs.mean()), 1)] + [round(x, 1) for x in _boot(fs, 4)[1:]]},
        "diff_switchyard_minus_bygari": {"mean": round(d, 1), "ci": [round(d_lo, 1), round(d_hi, 1)],
                                         "indistinguishable": bool(d_lo <= 0 <= d_hi)},
        "crossover_txn": crossover,
        "explore_cost_per_seed": round(float(np.mean([r["explore_cost"] for r in res])) / 100.0, 1),
        "H3_starved_region": {
            "traffic_share": round(starved_share, 4),
            "restricted_diff_large": {"mean": round(ld, 1), "ci": [round(ld_lo, 1), round(ld_hi, 1)],
                                      "indistinguishable": bool(ld_lo <= 0 <= ld_hi)},
            "remainder_diff_small": {"mean": round(sd, 1), "ci": [round(sd_lo, 1), round(sd_hi, 1)],
                                     "indistinguishable": bool(sd_lo <= 0 <= sd_hi)},
            "final_starved_share_large_to_paapc": {
                "bygari": round(float(starved_b[:, -1].mean()), 4),
                "switchyard": round(float(starved_s[:, -1].mean()), 4)}},
    }


def make_plots(cell03: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = cell03["grid"]
    mb = np.array(cell03["cum_b_mean"]); ms = np.array(cell03["cum_s_mean"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, mb, color="#c65a2e", lw=2, label="bygari_baseline (cumulative)")
    ax.plot(grid, ms, color="#2f6db3", lw=2, label="switchyard, net of exploration")
    ax.set_xlabel("transactions processed (per seed)")
    ax.set_ylabel("cumulative net revenue (₹, mean over 40 seeds)")
    ax.set_title("Cumulative net revenue — bygari_baseline vs switchyard (ε=0.03, 40 seeds)")
    ax2 = ax.twinx()
    ax2.plot(grid, ms - mb, color="#4c9a52", ls="--", lw=1.5, label="switchyard − bygari (right)")
    ax2.axhline(0, color="#888", lw=0.8)
    ax2.set_ylabel("difference (₹): switchyard − bygari", color="#4c9a52")
    ax2.tick_params(axis="y", labelcolor="#4c9a52")
    dm = cell03["diff_switchyard_minus_bygari"]
    tag = "indistinguishable (CI crosses 0)" if dm["indistinguishable"] else "distinguishable"
    ax2.annotate(f"final diff ₹{dm['mean']:,.0f} — {tag}\ncrossover: {cell03['crossover_txn']}",
                 xy=(grid[len(grid)//3], (ms - mb).min()), fontsize=9, color="#2a6a30")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=8); ax.grid(True, alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(ART, "cumulative_value.png"), dpi=110, metadata={"Software": "switchyard"})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, cell03["starved_b_mean"], color="#c65a2e", lw=2, label="bygari_baseline")
    ax.plot(grid, cell03["starved_s_mean"], color="#2f6db3", lw=2, label="switchyard")
    ax.set_xlabel("transactions processed (per seed)")
    ax.set_ylabel("share of large-ticket (>₹5k) traffic sent to pa/pc")
    ax.set_title("Traffic to the starved large-ticket options (ε=0.03, 40 seeds)")
    ax.legend(); ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(os.path.join(ART, "starved_region_traffic.png"), dpi=110, metadata={"Software": "switchyard"})
    plt.close(fig)


def run(save=False) -> dict:
    """H1 ε-sweep at 40 seeds, then H2 (best ε, 2M/seed, 10 seeds).

    When ``save`` is set, the partial result is flushed to timeseries.json after
    each ε-cell and after H2, so a late failure never discards completed compute.
    """
    r: dict = {"H1_eps_sweep": {}, "best_eps_for_switchyard": None, "H2_horizon_2M": None}
    for e in EPS_GRID:
        r["H1_eps_sweep"][f"{e:.2f}"] = run_cell(e, SEEDS, N_PER_SEED)
        if save:
            _save(r)
    best_eps = max(EPS_GRID, key=lambda e: r["H1_eps_sweep"][f"{e:.2f}"]["diff_switchyard_minus_bygari"]["mean"])
    r["best_eps_for_switchyard"] = best_eps
    r["H2_horizon_2M"] = run_cell(best_eps, list(range(2000, 2010)), 2_000_000)
    if save:
        _save(r)
    return r


def main():
    r = run(save=True)
    make_plots(r["H1_eps_sweep"]["0.03"])
    print(f"{'ε':>5s} {'bygari':>12s} {'switchyard':>12s} {'diff':>10s} {'CI':>22s} {'xover':>8s} {'H3 large diff (CI)':>26s}")
    for e in EPS_GRID:
        c = r["H1_eps_sweep"][f"{e:.2f}"]; d = c["diff_switchyard_minus_bygari"]; h3 = c["H3_starved_region"]["restricted_diff_large"]
        tag = " *indist*" if d["indistinguishable"] else ""
        print(f"{e:>5.2f} {c['final']['bygari'][0]:>12.1f} {c['final']['switchyard'][0]:>12.1f} "
              f"{d['mean']:>10.1f} {str(d['ci']):>22s} {str(c['crossover_txn']):>8s} "
              f"{h3['mean']:>10.1f} {str(h3['ci']):>15s}{tag}")
    h2 = r["H2_horizon_2M"]
    print(f"\nH2 (ε={r['best_eps_for_switchyard']}, 2M/seed × {len(h2['seeds'])} seeds): "
          f"diff {h2['diff_switchyard_minus_bygari']['mean']} CI {h2['diff_switchyard_minus_bygari']['ci']} "
          f"crossover {h2['crossover_txn']}")
    print(f"starved-region traffic share: {r['H1_eps_sweep']['0.03']['H3_starved_region']['traffic_share']}")


if __name__ == "__main__":
    main()
