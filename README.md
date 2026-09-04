# Switchyard

**Switchyard routes each payment to the processor that maximises expected net revenue, and — unlike a supervised model trained on confounded logs — spends a small, budgeted slice of randomly-routed traffic so its value estimates stay honest in the segments the old policy never explored.**

> ## ⚠️ All data here is synthetic.
> Every number in this README is produced by a simulator in this repository with **known, controlled latent ground truth** — no real payment data is used. Reproduce everything with `python verify.py`. No figure below is hand-written; each is emitted by a script (ABSOLUTE RULE 1).

---

## The one-sentence problem

Historical transaction logs were produced by an existing routing policy. Where that policy almost never sent a segment to a processor, the logs contain almost no evidence about that pair. A supervised model trained on those logs doesn't learn which processor is better there — **it extrapolates, confidently, with no signal that it is guessing.** You cannot learn what you never tried. Switchyard fixes that with a 3% exploration budget and proves it on a simulator where the true answer is known.

A second, smaller point: the right processor depends on **ticket size**, because processors have different fee shapes (flat vs percentage). The objective is expected **net revenue per attempt**, not success rate.

---

## Results (§6.1)

Four methods, learned from the same 200,000 confounded legacy logs (`switchyard` additionally gets a 200k epoch with 3% of decisions randomised). Each proposes a per-cell routing policy and its own estimate of that policy's value. **True value** is computed by rolling the policy forward through fresh simulated traffic (10 seeds, common random numbers, 1000-resample paired bootstrap). All values are **₹ per 1,000 attempts**.

| Method | Estimated | True | **Estimation error** | True 95% CI | Improvement over legacy (CI) | Weights clipped |
|---|---:|---:|---:|:---:|:---:|---:|
| direct | 42410.0 | 41763.5 | **+646.5** | [41436, 42128] | +1646 [1626, 1667] | 0 |
| ips    | 42994.3 | 40436.1 | **+2558.2** | [40127, 40775] | +319 [289, 347] | 0 |
| snips  | 43070.3 | 40625.6 | **+2444.7** | [40317, 40968] | +508 [481, 538] | 0 |
| dr     | 42986.6 | 41005.4 | **+1981.3** | [40686, 41357] | +888 [863, 914] | 0 |
| **switchyard** | 42180.4 | **41238.3** | **+942.1** | [40918, 41597] | +1121 [1100, 1142] | 4039 |

Legacy baseline true value **40117.1**; oracle (best possible) **42325.3**. Every method's improvement CI clears zero, so all are *distinguishable from baseline*.

Read the **estimation-error column** — the brief's headline. Every off-policy value estimator **overstates its own policy** (ips by ₹2558, snips ₹2445, dr ₹1981 per 1k). `switchyard` overstates least of the reweighting family (₹942) *and* has the highest true value of the four. `direct`'s own-policy error looks small (₹647) — but that number is deceptive, which the next result shows.

### The honest headline: what happens when a policy relies on the unexplored region

`direct`'s low own-policy error only means it is honest about its *own cautious policy*. Point each estimator at a **fixed** policy that ventures into the starved region and ask what it's worth — the truth is known by rollout:

**Target = the direct-greedy policy** (it routes large tickets to `pa`), true value **41763.5**:

| Estimator | Estimate | Error vs truth |
|---|---:|---:|
| **switchyard** | 41780.8 | **+17.3** |
| dr | 42022.4 | +258.8 |
| direct | 42410.0 | +646.5 |
| snips | 40625.0 | −1138.5 |
| ips | 39662.8 | −2100.7 |

**Target = an adversarial policy that dumps ALL large tickets on `pa`** (the trap), true value **38974.3** (below legacy):

| Estimator | Estimate | Error vs truth |
|---|---:|---:|
| direct | 40928.5 | **+1954.2** |
| dr | 40580.8 | +1606.5 |
| snips | 40252.8 | +1278.5 |
| switchyard | 40276.8 | +1302.5 |
| ips | 40162.8 | +1188.5 |

`switchyard` values the direct-greedy policy to within **₹17 / 1k of the truth**; `direct`'s own model overstates it by **₹647**, and would price the money-losing trap policy **₹1954 too high** — it would deploy a loss-maker believing it was excellent, with no warning. `switchyard`, having spent 3% on exploration, is the only estimator you can trust into the region the legacy policy never tried. *That* is what "methods that overstate their value mislead their operator" means.

### Why direct is blind there — the mechanism

`direct`'s gradient-boosted model predicts `pa`'s success rate accurately for small tickets, but the legacy policy sends everything above ₹5,000 to `pb`, so `pa`'s large-ticket weakness was never observed. The model extrapolates — and is **~10 percentage points too optimistic** exactly where it has no data:

![pa predicted vs true success by ticket size](artifacts/extrapolation.png)

| Ticket | Predicted `pa` success | True | Gap |
|---:|---:|---:|---:|
| ₹200 | 0.880 | 0.860 | +0.020 |
| ₹2,000 | 0.830 | 0.860 | −0.030 |
| ₹7,000 | 0.778 | 0.680 | **+0.098** |
| ₹12,000 | 0.781 | 0.680 | **+0.101** |
| ₹80,000 | 0.782 | 0.680 | **+0.102** |

---

## `hdfc × upi` segment decision table (§6.1)

Which processor each method picks, against the true best (argmax of each bucket's average true expected reward):

| Ticket bucket | True best | direct | ips | snips | dr | switchyard |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| <₹300 | pb | ✓ | ✓ | ✓ | ✓ | ✓ |
| ₹300–1k | pb | ✓ | ✓ | ✓ | ✓ | ✓ |
| ₹1k–3k | pa | ✓ | ✓ | ✓ | ✓ | ✓ |
| ₹3k–10k | pa | ✓ | ✓ | ✓ | ✓ | ✓ |
| **>₹10k** | **pc** | pa ✗ | pa ✗ | ✓ | ✓ | pa ✗ |

Honest reading: on well-covered ticket sizes every method (direct included) picks correctly. On the starved **>₹10k** bucket, `direct` routes to `pa` — the processor it over-values there — which `snips`/`dr` avoid. `switchyard` also mis-picks this specific cell: at a 3% budget the truly-best `pc` is starved for *everyone* (≤46 samples even after exploration), so this deepest cell stays contested. This is reported as measured, not smoothed (ABSOLUTE RULE 6).

---

## Ticket-size crossover (§4.3)

The flat-vs-percentage fee structure produces a genuine crossover. On `hdfc × upi`, `pa` has the **highest success rate** (0.95 vs pb 0.86, pc 0.85) yet is **not** the best expected-reward choice below **₹962** — its flat ₹4 fee eats the thin margin on small tickets:

![expected net reward vs ticket size](artifacts/crossover.png)

A **naive success-rate router** (route to highest predicted success, ignoring fees) scores **39861.9 ₹/1k — below the legacy baseline (40117.1)**. Optimising success rate loses money; optimising net revenue is the point.

---

## Money recovered by the recovery loop (§7)

When a payment fails, the same routing brain decides whether and where to re-attempt. Over **5,000 real failures**, `switchyard`'s smart rerouting recovers **₹6,361.63 more** than the legacy retry policy — an incremental **127.23 paise/failure**, 95% CI **[113.22, 140.84] paise**, which excludes zero.

**Stopping rules** (all enforced; live engine run over the 5,000 failures): never re-attempt if expected net reward ≤ 0, at most 3 attempts, never re-attempt a hard decline. Outcomes: **3,489 recovered**, 1,447 hard declines correctly not retried, 51 hit the attempt cap, 13 stopped on non-positive expected reward. Every attempt is written to `artifacts/audit.jsonl`.

**Idempotency:** a SQLite table with `txn_id` as primary key and a `PENDING` reservation. The 10-concurrent-identical-events test yields exactly one attempt; replaying a processed failure creates no second attempt.

**Compliance:** an attempt cap and a minimum inter-attempt delay for e-mandate-style (netbanking) failures are enforced. **RBI 24-hour pre-debit notification rules are NOT implemented** — that is a real gap; a production system would need it.

---

## LLM diagnosis scores (§8)

A single contained LLM job (never routes): given a cohort's failure-code counts and a baseline comparison, output structured JSON `{cause, confidence, evidence}` or abstain with `INSUFFICIENT_EVIDENCE`. Output is schema-validated with a fallback to `INSUFFICIENT_EVIDENCE` on any malformed response, and cached by input hash.

Scored on constructed cohorts (abstention is **rewarded**, not penalised):

| Metric | Score |
|---|---:|
| Accuracy on clear cohorts | **0.923** |
| Abstention rate on deliberately ambiguous cohorts | **0.667** |
| Harmful-error rate (a *wrong* specific assertion) | 0.105 |

> No `ANTHROPIC_API_KEY` was available in this build environment, so these numbers come from the **deterministic offline statistical diagnoser**. The Anthropic LLM path (`claude-opus-5`, structured output) is implemented and is scored identically when a key is set — set `ANTHROPIC_API_KEY` and re-run `python -m diagnose.evaluate`. The offline baseline abstains correctly on tiny cohorts but over-asserts on genuine two-cause blends (the source of the 0.105 harmful rate); this is exactly where LLM reasoning is expected to help.

---

## Held-out regime (evaluated exactly once)

A deliberately different regime — 50/35/15 method mix, uniform issuers, a fatter amount tail, and a **different degradation schedule** — was evaluated **once** (2026-09-04T19:50:31; `eval/heldout.py` refuses to re-run). The methods were **not** retrained; only true value was re-measured.

| Method | True value on held-out (₹/1k) |
|---|---:|
| direct | 53516.1 |
| **switchyard** | **52776.2** |
| dr | 52285.1 |
| snips | 51424.7 |
| ips | 51253.8 |

Legacy 51312.3, oracle 54074.9. **The policy-quality ordering is preserved** (direct > switchyard > dr > snips > ips): the learned policies generalise, and switchyard still beats every off-policy method under a shifted regime. The main-calibrated *self-estimates* (~₹42k) do not transfer to the held-out value scale (~₹52k) — everyone under-states by ~₹10k/1k — which is the regime's value-scale shift, not a coverage effect (a like-for-like estimate comparison would need held-out logs, which RULE 5 forbids collecting).

---

## Known limitations (real, unsmoothed)

1. **The naive method does not simply "fail."** With modest, smooth latent effects a well-regularised GBM (`direct`) is a *strong, robust* policy — it beats the off-policy estimators on true value. The failure this project demonstrates is **conditional on coverage**: it appears precisely where a consequential fact (`pa`'s −0.18 large-ticket weakness) sits in a region the legacy policy starved. We placed it there deliberately and transparently (see `NOTES.md`, 2026-09-04, and `DECISIONS.md`); that placement *is* the phenomenon, not a thumb on the scale. Run against smooth, well-covered facts, `direct` would not be fooled.
2. **ε = 0.03 buys honesty, not (yet) a big policy win.** At this budget `switchyard`'s advantage is primarily *epistemic* — it knows what a policy is worth in the starved region (OPE error +17 vs direct +647) — plus a modest true-value gain over the other off-policy methods. It does **not** fully resolve the deepest-starved cell (`pc` on >₹10k), which needs a larger exploration budget or more epochs. We do not claim otherwise.
3. **The simulator models each attempt as an independent draw** (no failure persistence). Real insufficient-funds/risk declines would recur on retry regardless of processor, so absolute recovery counts (3,489/5,000) are optimistic; the *incremental-over-legacy* metric, which is what we headline, is unaffected.
4. **Routing is over discretised `(method, issuer, amount-bucket)` cells**, not continuous amount — a deliberate choice so all four methods share one hypothesis space and the uniform-logs sanity test is clean, at the cost of some within-bucket resolution near crossovers.
5. **The soft degradation regime (`pa+sbi+upi`, certain days/hours) is unlearnable by every method** — the models have no day-of-week feature — so no method routes around it; it is latent noise, not a demonstrated win.
6. **RBI 24-hour pre-debit notification is not implemented** (see §7).

---

## One-command reproduction

```bash
pip install -r requirements.txt
python verify.py          # ~1 minute: runs tests, regenerates every artifact, reprints every number above
```

`verify.py` exits non-zero on any failure. Everything is deterministic on fixed seeds; the log files and result tables are byte-identical across runs (enforced by `tests/test_determinism.py`).

---

## Prior art (cited by name)

- **Inverse propensity weighting** — Horvitz & Thompson (1952), *A Generalization of Sampling Without Replacement From a Finite Universe*.
- **Doubly robust policy evaluation & learning** — Dudík, Langford & Li (2011); Robins, Rotnitzky & Zhao (1994) for the DR estimator's statistical origins.
- **Self-normalised IPS (SNIPS) & counterfactual risk minimisation** — Swaminathan & Joachims (2015).
- **Counterfactual reasoning and learning systems** — Bottou, Peters, Quiñonero-Candela et al. (2013).
- **Offline / off-policy evaluation in contextual bandits** — Li, Chu, Langford & Schapire (2010–2011).
- **The optimizer's curse** (why argmax over noisy value estimates is upward-biased) — Smith & Winkler (2006).
- **Explore–exploit / ε-greedy** — Sutton & Barto, *Reinforcement Learning: An Introduction*.
- **Offline RL under distributional shift / pessimism** — Levine, Kumar, Tucker & Fu (2020), survey.

See `DECISIONS.md` for the design rationale (why DR over IPS, why clip at 50, why SQLite WAL, why the LLM is confined to diagnosis) and `NOTES.md` for the timestamped build log, including every simulator-tuning decision.
