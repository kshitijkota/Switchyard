# NOTES — live build log

Timestamped, append-only. Every bug, tuning decision, and open question goes
here as it happens (AGENT_BRIEF §11, §12.4).

---

### 2026-09-04 18:25 IST — Project setup

- The project is named **Chowk** in the brief; the git repository is
  **Switchyard** (remote `origin` = github.com/kshitijkota/Switchyard.git,
  fresh `main`, no commits). Decision: build the whole project inside the
  Switchyard repo and commit to `main`. `AGENT_BRIEF.md` lives one directory up
  and is treated as read-only spec, never modified.
- Toolchain present: Python 3.12, numpy 2.3.1, scikit-learn 1.7.2,
  matplotlib 3.9.2. No system installs needed.
- Using the machine's existing global git identity for commits; global git
  config is NOT modified (§0).

### 2026-09-04 18:25 IST — Architecture interpretation (estimators & policies)

The brief describes `ips`/`dr`/`chowk` mostly as value *estimators* but also
requires all four to be *policies* with a `recommend()` and an "estimated value"
row. Interpretation chosen (recorded because it shapes §6):

- **Unified per-cell design.** Context is discretised into cells
  `(method, issuer, amount_bucket)`. For each `(cell, processor)` each method
  produces an estimate of expected net reward. The method's **policy** is the
  per-cell argmax; its **estimated value** is that policy's value under its own
  estimator; **true value** comes from rolling the policy forward through fresh
  simulated traffic (only the eval/sim side may see ground truth).
- Amount buckets are cut at economically meaningful paise boundaries (the
  fee crossovers and the ₹10k interaction), listed in `estimators/segments.py`.
- This keeps a single shared interface across all four methods and makes the
  "uniform-random logs ⇒ all estimators agree" sanity test (§9) clean.

### 2026-09-04 18:25 IST — Hypothesis for HOW `direct` fails (to verify, not assume)

The brief asserts `direct` should pick wrong on `hdfc × upi` and `chowk` right,
and explicitly allows that this may need tuning of effect sizes/coverage — with
every tuning pass logged here (§6.1, §12.4). My working hypothesis before seeing
any number:

- `pa + hdfc + upi` gets the **+0.07** success boost, so `pa` has the highest
  *success rate* there and is heavily covered by the legacy policy (upi ⇒ pa 0.90).
- But the objective is **expected net reward**, not success. Because `pa` charges
  a flat ₹4 while `pb`/`pc` charge a percentage, at **small ticket sizes** the
  flat fee dominates and `pb`/`pc` can be the truly better choice (the crossover).
- The legacy policy almost never sends upi to `pb` (0.03), so `direct` has little
  evidence about `pb` on upi and extrapolates — "you cannot learn what you never
  tried." `chowk`'s ε-exploration slice restores that coverage.

I will BUILD the machinery honestly, then RUN it and report whatever actually
happens. If the headline does not reproduce it will be tuned (logged here) or
reported as a negative result (§12.6) — never fabricated.

### Open questions

- None blocking yet.
