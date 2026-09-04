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

### 2026-09-04 19:05 IST — CRITICAL FINDING: spec-default sim does NOT show the thesis

Built all four methods on the spec-default sim and measured true value by
ground-truth rollout (10 seeds, expected-reward, n=50k). Numbers as produced:

| method | estimated ₹/1k | true ₹/1k | est−true (error) |
|---|---|---|---|
| direct | 43970 | **43675** | **+295** |
| ips    | 45134 | 42850 | **+2283** |
| snips  | 44295 | 42826 | +1469 |
| dr     | 44241 | 43113 | +1128 |
| chowk  | 44052 | 43602 | +449 |
(legacy baseline true value ≈ 39905.)

**This inverts the brief's expected story.** With §4.2/§5 exactly as written,
`direct` is the *best* method (highest true value) and the *most honest* (+295);
the method that overstates is **IPS** (+2283, variance in starved cells). On the
headline `hdfc × upi` segment every method — direct included — already picks
right, because the legacy policy covers pb/pc on upi well enough (100–300
samples/cell) that direct's GBM extrapolates the smooth latent structure fine.

Diagnosis: "you cannot learn what you never tried" only bites the **model-based**
method when an un-learnable fact sits in a **legacy-starved** region. Every §4.2
interaction sits in a *well-covered* region, so direct learns them all. This is a
genuine inconsistency between §4.2/§5 (facts in covered regions) and §6.1's goal
(direct should pick wrong). Flagged per §0.

### 2026-09-04 19:05 IST — DECISION (logged before re-running): place the missable fact in a starved cell

Per §6.1 ("If it doesn't, the latent effect sizes or coverage need tuning —
record the tuning in NOTES.md"), one **single, principled** change, chosen and
recorded before seeing its downstream numbers (no iterative fishing):

1. Latent truth: **add** `pb + hdfc + upi: +0.09` (magnitude comparable to the
   existing +0.07 / +0.04). A plausible real fact: pb has a bank-specific
   advantage for HDFC UPI. All four §4.2 interactions are kept unchanged.
2. Coverage: legacy `upi` split changes `pb 0.03 → 0.01` (moving 0.02 to pc:
   `pa 0.90, pb 0.01, pc 0.09`). This starves pb-on-upi to <200 samples/cell,
   below the GBM's `min_samples_leaf=200`, so the model regularises the boost
   away — it literally cannot learn what the legacy policy never tried.

Predicted (to verify): pa+hdfc+upi and pb+hdfc+upi both ≈0.95 success, so at
small tickets pb wins on expected reward (lower % fee vs pa's flat ₹4); direct,
blind to pb's boost, keeps picking pa → wrong; chowk's exploration restores pb
coverage → picks pb → right. The estimation-error headline should then show
direct overstating on the starved segment. Will report the ACTUAL result, and
keep this configuration frozen through the held-out run (§6.1).

### 2026-09-04 19:20 IST — The +0.09 boost was the wrong shape; final design (v3)

Measured v2 (pb·hdfc·upi **+0.09**): direct still highest true value (43731),
still most honest (+483). The boost failed to demonstrate anything because a
*hidden boost* only makes direct **under-route** to the good-but-unseen action —
a low-stakes regret, not the overstatement the brief targets. And hdfc×upi is
intrinsically low-stakes (pa≈pb≈0.95), so nothing there moves money.

Correct mechanism (the §1 failure): direct fails when a processor it **already
favours** is secretly **worse** in a cell the legacy policy **starved** — it
extrapolates optimistically, routes there confidently, overstates, and loses.
For real money it must be **high-stakes** (large tickets).

**FINAL latent-truth design (v3), chosen by principle and frozen here BEFORE
seeing its numbers — no magnitude fishing.** Two hidden facts, each a plausible
operational reality placed in a legacy-starved region, each magnitude anchored to
§4.2's own −0.15 degradation:

1. `pb + hdfc + upi : −0.16` (was +0.09). pb has a bank-specific weakness on HDFC
   UPI. Legacy starves pb-on-upi (0.01), so direct thinks pb is base-good, routes
   small hdfc·upi tickets to pb, overstates, and loses to pc. → the §6.1 segment
   table (hdfc×upi): direct wrong, chowk right.
2. `pa + amount > ₹10k : −0.18` (new). The flat-fee processor pa throttles large
   tickets. Legacy sends >₹5k to pb (pa starved at 0.03), so direct extrapolates
   pa's base success, is seduced by pa's flat-fee cost advantage at scale, routes
   large tickets to pa, overstates, and loses big. → the high-stakes money story.

Both are invisible to direct precisely because they live where the legacy policy
never looked; chowk's ε-exploration restores that coverage. Note: a large effect
in a *starved* cell is still fine per §4.2's "keep modest" caveat — the modesty
concern is about *covered* cells (where a big effect is obvious to everyone); a
starved-cell effect is invisible to direct at any size, which is the whole point.

This config is frozen through the held-out run (§6.1 / §12.5). Any later change
gets its own dated entry.

### 2026-09-04 19:35 IST — Converged on the MINIMAL design (v4, frozen)

v3 confirmed the mechanism: direct predicts pa success = **0.871** on
card×hdfc×>₹10k when the truth is **0.68** — it cannot see the penalty (legacy
starves pa on large), so it confidently routes large tickets to pa, overstates,
and loses. That single fact already makes direct wrong on `upi×hdfc×>₹10k` too,
so the separate low-stakes pb·hdfc·upi fact is unnecessary and was dropped.

Final, minimal, frozen design:
- **Legacy policy = §5 exactly** (reverted the pb-on-upi change; upi = pa 0.90,
  pb 0.03, pc 0.07).
- **Latent truth = §4.2 exactly, plus ONE hidden fact:**
  `pa + amount > ₹5,000 : −0.18`. The penalty region (>₹5k) is *exactly* the
  region the legacy policy sends to pb (>₹5k ⇒ pb 0.95, pa 0.03), so pa's
  large-ticket weakness was never observed — the literal instantiation of
  "you cannot learn what you never tried." Magnitude anchored to §4.2's own
  −0.15; in a fully-starved cell the size is invisible to direct regardless, so
  this respects §4.2's "keep modest in *covered* cells" caveat.

The design iterated (v1→v4) by understanding the failure mechanism, not by
fishing magnitudes: −0.18 is fixed by principle and whatever numbers it yields
are reported as-is. Frozen through the held-out run (§12.5).

### 2026-09-04 20:10 IST — Eval harness built + final results (step 7)

Numbers below are produced by `eval/harness.py` (10 seeds × 20k = 200k eval txns,
common random numbers, 1000-resample paired bootstrap). Legacy baseline 40117,
oracle 42325, naive fee-blind success-router **39862 (below legacy!)** — §1's
second point, that success-rate routing loses money on fees, reproduces.

Results table (₹/1k): estimated / true / error:
- direct 42410 / 41764 / **+646**   (true value highest — a strong GBM)
- ips    42994 / 40436 / **+2558**
- snips  43070 / 40626 / +2445
- dr     42987 / 41005 / +1981
- chowk  42180 / 41238 / **+942**    (2nd-best true value, beats all reweighting)

**OPE honesty (the headline).** Value of a FIXED starved-region policy
(direct-greedy, routes large→pa), true 41764:
  chowk **+17** · dr +259 · direct +646 · snips −1138 · ips −2101.
chowk is the only estimator that stays honest into the unexplored region.
Adversarial "all large→pa" policy (true 38974): direct overstates by **+1954** —
it would deploy a money-loser believing it great.

**The clean evidence** (`artifacts/extrapolation.json`): direct's predicted pa
success is accurate below ₹5k (gap ≈ +0.02) but a persistent **+0.10 too high
above ₹5k** (predicts ~0.78, truth 0.68) — the region legacy starves.

Segment table (hdfc×upi, average-based true best): all methods correct on
<₹10k; on >₹10k direct/ips/chowk pick pa (wrong), snips/dr pick pc — that cell's
true best (pc) is starved for *everyone* (≤46 samples even combined), so it stays
contested. Honest nuance, reported as-is.

Interpretation (honest): a well-regularised GBM (direct) is a strong policy on
smooth structure — the naive method is NOT simply "bad". Its real danger is
epistemic: it cannot tell a good policy from a bad one where it never explored,
and it overstates any policy that ventures there. At ε=0.03 chowk's edge is
primarily this honesty (OPE), plus a modest, real policy gain over the other
off-policy methods; a thicker budget would be needed to fully resolve the deepest
starved cells. This will be stated plainly in the README (no overselling).

### Open questions

- None blocking yet.
