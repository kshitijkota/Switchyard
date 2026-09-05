#!/usr/bin/env python3
"""verify.py — AGENT_BRIEF §10.

One command: run the test suite, regenerate every artifact, reprint every number
that appears in the README, and exit non-zero on any failure.

    python verify.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

FAILURES: list[str] = []


def step(name: str):
    print(f"\n{'='*72}\n▶ {name}\n{'='*72}")


def guard(name: str, fn):
    try:
        t = time.time()
        out = fn()
        print(f"  ✓ {name}  ({time.time()-t:.1f}s)")
        return out
    except Exception as e:  # noqa: BLE001
        FAILURES.append(f"{name}: {type(e).__name__}: {e}")
        print(f"  ✗ {name}: {type(e).__name__}: {e}")
        return None


def run_tests() -> bool:
    step("1. Test suite (pytest)")
    env = {**os.environ, "PYTHONPATH": ROOT}
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=ROOT, env=env)
    ok = r.returncode == 0
    if not ok:
        FAILURES.append("pytest: test suite failed")
    return ok


def main() -> int:
    run_tests()

    step("2. Regenerate simulator data")
    from sim.generate_logs import generate as gen_logs
    from sim.explore_slice import generate as gen_explore
    log_stats = guard("200k confounded logs", gen_logs)
    exp_stats = guard("200k exploration slice", gen_explore)

    step("3. Regenerate artifacts")
    from sim.plots import make_crossover_plot
    from eval.harness import run_evaluation, RESULTS_PATH, print_results
    from recovery.evaluate import evaluate as rec_eval
    from diagnose.evaluate import evaluate as diag_eval, RESULTS_PATH as DIAG_PATH, STATISTICAL_REF_PATH as DIAG_STAT

    crossover = guard("crossover plot", make_crossover_plot)
    results = guard("evaluation harness", run_evaluation)
    if results:
        with open(RESULTS_PATH, "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)

    from eval.exploration_curve import run as expl_run, make_plot as expl_plot, JSON_PATH as CURVE_JSON

    def _expl():
        r = expl_run()
        with open(CURVE_JSON, "w") as fh:
            json.dump(r, fh, indent=2, sort_keys=True)
        expl_plot(r)
        return r

    def _diag():
        r = diag_eval()
        with open(DIAG_PATH, "w") as fh:
            json.dump(r, fh, indent=2, sort_keys=True)
        return r

    def _diag_stat():
        r = diag_eval(force_provider="statistical")
        with open(DIAG_STAT, "w") as fh:
            json.dump(r, fh, indent=2, sort_keys=True)
        return r

    from diagnose.threeway import evaluate as threeway_eval, RESULTS_PATH as TW_PATH

    def _threeway():
        r = threeway_eval()
        with open(TW_PATH, "w") as fh:
            json.dump(r, fh, indent=2, sort_keys=True)
        return r

    curve = guard("exploration price curve", _expl)

    from eval.live_experiment import run_cell as live_cell, make_plots as live_plots, SEEDS, N_PER_SEED

    # The full diagnostic grid (H1 ε-sweep at 40 seeds + H2 2M/seed) is a long
    # run committed to artifacts/timeseries.json (RULE 5-style: expensive, run
    # once). verify re-runs only the headline cell (ε=0.03, 40 seeds, 50k/seed)
    # to reprint the live numbers, and reads the committed grid for the rest.
    def _live():
        c = live_cell(0.03, SEEDS, N_PER_SEED)
        live_plots(c)
        return c

    live = guard("live online race — headline cell ε=0.03, 40 seeds (TASK A)", _live)
    live_grid = None
    _tp = os.path.join(ROOT, "artifacts", "timeseries.json")
    if os.path.exists(_tp):
        with open(_tp) as fh:
            live_grid = json.load(fh)
    recovery = guard("recovery evaluation", rec_eval)
    diagnosis = guard("diagnosis (gemini/committed cache)", _diag)
    diagnosis_stat = guard("diagnosis statistical reference", _diag_stat)
    threeway = guard("open-world three-way (rule/trained/LLM × closed/open)", _threeway)

    # held-out regime is run ONCE (RULE 5); read the committed artifact, never re-run.
    heldout = None
    _hp = os.path.join(ROOT, "artifacts", "heldout_results.json")
    if os.path.exists(_hp):
        with open(_hp) as fh:
            heldout = json.load(fh)

    # ---- Reprint every number that appears in the README --------------------
    step("4. README numbers (single source of truth)")
    if log_stats:
        print(f"\n[data] 200k logs — success rate {log_stats['success_rate']}, "
              f"processor share {log_stats['processor_share']}")
    if exp_stats:
        print(f"[data] exploration slice — explore rate {exp_stats['explore_rate']}")
    if crossover:
        cx = crossover["crossovers"]
        print(f"\n[§4.3 crossover] hdfc×upi success {crossover['success_prob']}; "
              f"crossovers {[(c['from_processor']+'→'+c['to_processor'], c['amount_rupees']) for c in cx]}")
    if results:
        print()
        print_results(results)
    if curve:
        print("\n[TASK 3 exploration price curve] ε: err_adv / err_starved / cost per 1k / true value")
        for p in curve["points"]:
            print(f"  ε={p['epsilon']:.2f}  err_adv={p['estimation_error_adversarial']['error']:.1f}  "
                  f"err_starved={p['estimation_error_starved']['error']:.1f}  "
                  f"cost={p['exploration_cost_per_1k']['value']:.1f}  "
                  f"true={p['true_policy_value']['value']:.1f}")
    if live:
        d = live["diff_switchyard_minus_bygari"]; h3 = live["H3_starved_region"]
        fs = live["H3_starved_region"]["final_starved_share_large_to_paapc"]
        print(f"\n[TASK A live race — ε=0.03, {len(live['seeds'])} seeds × {live['n_per_seed']:,}/seed]")
        print(f"  final net ₹/seed — bygari {live['final']['bygari'][0]} vs switchyard {live['final']['switchyard'][0]}")
        print(f"  switchyard − bygari  {d['mean']} CI {d['ci']}  "
              f"({'INDISTINGUISHABLE — CI crosses 0' if d['indistinguishable'] else 'distinguishable'}); "
              f"crossover {live['crossover_txn']}")
        print(f"  [H3] starved large-ticket (>₹5k) traffic share {h3['traffic_share']}; "
              f"restricted diff (large) {h3['restricted_diff_large']['mean']} CI {h3['restricted_diff_large']['ci']}; "
              f"remainder diff (small) {h3['remainder_diff_small']['mean']} CI {h3['remainder_diff_small']['ci']}")
        print(f"  [H3] large-ticket share routed to pa/pc — bygari {fs['bygari']} vs switchyard {fs['switchyard']}")
    if live_grid:
        h1 = live_grid["H1_eps_sweep"]; h2 = live_grid["H2_horizon_2M"]
        print(f"\n[TASK A full grid — read from committed artifacts/timeseries.json]")
        print(f"  H1 ε-sweep (40 seeds × 50k): ε → switchyard−bygari CI (xover)")
        for e in sorted(h1):
            c = h1[e]; dd = c["diff_switchyard_minus_bygari"]
            tag = " *indist*" if dd["indistinguishable"] else ""
            print(f"    ε={e}  diff {dd['mean']:>9.1f} CI {dd['ci']}  xover {c['crossover_txn']}{tag}")
        hd = h2["diff_switchyard_minus_bygari"]
        print(f"  H2 horizon (ε={live_grid['best_eps_for_switchyard']}, {h2['n_per_seed']:,}/seed × "
              f"{len(h2['seeds'])} seeds): diff {hd['mean']} CI {hd['ci']} "
              f"({'indistinguishable' if hd['indistinguishable'] else 'distinguishable'}); xover {h2['crossover_txn']}")
    if recovery:
        m = recovery["money_recovered"]
        print(f"\n[§7 money recovered] {m['n_failures']} failures: incremental "
              f"₹{m['incremental_total_rupees']} (per-failure {m['incremental_per_failure_paise']}¢, "
              f"CI {m['incremental_ci_per_failure_paise']}¢, "
              f"distinguishable={m['distinguishable_from_zero']})")
        le = recovery["live_engine"]
        print(f"[§7 engine] processed {le['n_processed']}, recovered {le['realized_recovered']}, "
              f"stops {le['stopped_reasons']}, dup-created-attempt={le['duplicate_replay_created_attempt']}")
    if diagnosis:
        print(f"\n[§8 diagnosis] measured_by={diagnosis['provider']} model={diagnosis['model']} "
              f"completed={diagnosis['cohorts_completed']}/{diagnosis['cohorts_total']}"
              f"{' (PARTIAL)' if diagnosis['partial_run'] else ''}")
        print(f"  accuracy_clear={diagnosis['accuracy_on_clear']} "
              f"abstention_ambiguous={diagnosis['abstention_rate_on_ambiguous']} (rewarded) "
              f"harmful={diagnosis['harmful_error_rate']} parse_failure={diagnosis['parse_failure_rate']} "
              f"tokens={diagnosis['total_input_tokens']}in/{diagnosis['total_output_tokens']}out "
              f"est_cost=${diagnosis.get('estimated_cost_usd', 0)}")
    if diagnosis_stat:
        print(f"[§8 diagnosis — offline statistical reference, NOT an LLM] "
              f"accuracy_clear={diagnosis_stat['accuracy_on_clear']} "
              f"abstention_ambiguous={diagnosis_stat['abstention_rate_on_ambiguous']} (rewarded) "
              f"harmful={diagnosis_stat['harmful_error_rate']}")
    if threeway:
        print("\n[TASK C open-world — accuracy on clear (harmful) by method × world]")
        for m in ("rule", "trained", "llm"):
            cw, ow = threeway["closed"][m], threeway["open"][m]
            print(f"    {m:8s}  closed {cw['accuracy_on_clear']} ({cw['harmful_error_rate']})   "
                  f"open {ow['accuracy_on_clear']} ({ow['harmful_error_rate']})")
        llm_c = threeway["closed"]["llm"]
        print(f"  [C3] LLM closed abstention incl-guardrail "
              f"{llm_c['abstention_on_ambiguous_incl_guardrail']}; model-own "
              f"{llm_c['model_own_abstention_rate']} (guardrail={llm_c['guardrail_abstentions']}, "
              f"model={llm_c['model_abstentions']})")
    if heldout:
        print("\n[§6.1 held-out regime — read from the run-once artifact, not re-run]")
        print(f"  legacy {heldout['legacy_baseline_value']}  oracle {heldout['oracle_value']} ₹/1k")
        for row in heldout["results_table"]:
            print(f"    {row['method']:11s} true_heldout={row['true_value_heldout']}")

    step("Summary")
    if FAILURES:
        print(f"✗ {len(FAILURES)} failure(s):")
        for f in FAILURES:
            print(f"   - {f}")
        return 1
    print("✓ all steps passed; all artifacts regenerated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
