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
    from diagnose.evaluate import evaluate as diag_eval

    crossover = guard("crossover plot", make_crossover_plot)
    results = guard("evaluation harness", run_evaluation)
    if results:
        with open(RESULTS_PATH, "w") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
    recovery = guard("recovery evaluation", rec_eval)
    diagnosis = guard("diagnosis evaluation", diag_eval)

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
        print(f"\n[§8 diagnosis] provider={diagnosis['provider']} "
              f"accuracy_clear={diagnosis['accuracy_on_clear']} "
              f"abstention_ambiguous={diagnosis['abstention_rate_on_ambiguous']} "
              f"harmful_error_rate={diagnosis['harmful_error_rate']}")

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
