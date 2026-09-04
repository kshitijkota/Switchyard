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

### 2026-09-04 18:34 IST — Simulator built (step 2)

Sanity numbers from the actual generators (200k, seed 42):
- amount: median ₹1,193 (target ~₹1,200), p99 ₹21,837, capped ₹200,000; 24.2%
  of traffic under ₹500, 12.5% over ₹5,000, 4.5% over ₹10,000.
- method 65/25/10, hdfc 24% (heavier), diurnal peaks at 11–13 and 19–21.
- scalar vs vectorised `success_prob` agree exactly (maxdiff 0).
- Mean success prob: pa 0.880, pb 0.866, pc 0.851 — modest spread as required.

Crossover (`artifacts/crossover.json`, computed analytically, not eyeballed):
for **hdfc × upi** the only argmax switch is **pb → pa at ₹962.03**. Below it pb
is the better *expected-reward* choice even though pa has the higher *success
rate* (0.95 vs 0.86) — the whole point of §1's second argument. Roughly 40% of
upi traffic falls below this crossover.

### 2026-09-04 18:34 IST — Decision: amount-bucket boundaries (for §6 cells)

Cells are `(method, issuer, amount_bucket)`. Boundaries (paise), chosen at
economically meaningful points — the fee crossovers above and the ₹10k
interaction — NOT tuned against any result:
`[0, 30k), [30k, 100k), [100k, 300k), [300k, 1_000k), [1_000k, ∞)`
i.e. `<₹300, ₹300–1k, ₹1k–3k, ₹3k–10k, >₹10k`. The ₹962 crossover sits on the
₹300–1k / ₹1k–3k boundary, so per-bucket argmax can separate "small ⇒ pb" from
"large ⇒ pa". Recorded here per §12.4.

### 2026-09-04 18:34 IST — Note on the "direct picks wrong" headline (pre-run)

Reasoned through the mechanics: `direct` optimises expected *reward* (not
success), so the crossover alone will not fool it if it learns success well.
Its failure, if it occurs, must come from **coverage** (§1 first argument):
biased success estimates in cells the legacy policy barely explored. Whether
that bias is large enough to flip an argmax is an empirical question the brief
explicitly flags as tunable/loggable (§6.1, §12.4). The **robust** headline is
the *estimation-error* column (§6.1): `direct`'s plug-in self-estimate is
optimistically biased regardless. Will measure both and report honestly.

### 2026-09-04 18:40 IST — Legacy policy + 200k logs (step 3)

- 200,000 txns generated in ~4s; success rate 0.877; processor share
  pa 0.55 / pb 0.15 / pc 0.30. Regime share: baseline 0.968, merchant_glitch
  0.019, issuer_degraded 0.008, network_incident 0.006 (small but each yields
  hundreds of failures for the §8 cohorts).
- Confounding confirmed (coverage share by segment): high-amount → pb 0.96;
  upi → pa 0.91, **pb 0.025** (near-zero); other → pc 0.85. So pb-on-upi and
  pa/pc-on-high-amount are the starved cells.
- Determinism verified: regenerating the same n twice is byte-identical for
  both files. Cause class appears ONLY in ground_truth.jsonl, never in logs.

### Open questions

- None blocking yet.
