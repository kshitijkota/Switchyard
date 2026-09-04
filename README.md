# Switchyard

**Switchyard routes each payment to the processor that maximises expected net revenue, and — unlike a supervised model trained on confounded logs — spends a small, budgeted slice of randomly-routed traffic so its value estimates stay honest in the segments the old policy never explored.**

> ## ⚠️ All data here is synthetic.
> Every number below is produced by a simulator in this repository with **known, controlled latent ground truth** — no real payment data is used, and no claim is made about any production system. Reproduce everything with `python verify.py`. No figure is hand-written; each is emitted by a script (ABSOLUTE RULE: no number appears in any document unless a script here produced it).

---

## Headline

Point each estimator at a routing policy that relies on the region the legacy policy never explored, and ask what it's worth — the truth is known by simulation.

- On an **adversarial policy that routes _all_ large tickets to `pa`** — a genuine money-loser, true value **₹38,974 / 1k** attempts, *below* the ₹40,117 legacy baseline — **`direct` values it ₹1,954/1k too high.** It would green-light the loser with no warning signal.
- On **`direct`'s own deployed greedy policy** (true value ₹41,763/1k), **`switchyard` is off by just ₹17.3/1k** — while `direct` overstates by ₹646.5 and plain IPS is off by ₹2,100.7. Switchyard is the only estimator you can trust into the unexplored region.

That honesty is bought by a 3% exploration budget, and the next section prices it.

---

## The exploration price curve

Sweep the exploration rate ε and measure, per ε, how honest the value estimates become and what the randomised traffic costs (bootstrap 95% CIs in `artifacts/exploration_curve.json`).

![exploration price curve](artifacts/exploration_curve.png)

| ε | estimation error, starved policy (₹/1k) | estimation error, adversarial policy (₹/1k) | exploration cost (₹/1k) | true policy value (₹/1k) |
|---:|---:|---:|---:|---:|
| 0.00 | +276.0 | +1609.0 | 0.0 | 41454.1 |
| 0.01 | +242.1 | +1311.0 | 18.0 | 41351.8 |
| **0.03** | **+17.3** | +1302.5 | 46.6 | 41238.3 |
| 0.10 | −254.0 | +1265.2 | 157.0 | 41335.9 |

**Where the curve stops being worth it:** the estimation error on the starved policy crosses honest (~0) at **ε ≈ 0.03**, then *over-corrects* to −₹254 at ε = 0.10 while the exploration cost more than triples (₹47 → ₹157/1k). So the knee is ε ≈ 0.03: near-zero error at modest cost, and nothing bought beyond it. The adversarial policy is starved so deeply (it forces *every* large ticket onto the unexplored `pa`) that even ε = 0.10 only trims its error from ₹1,609 to ₹1,265/1k — honest evaluation there would need a larger budget, and we say so rather than imply ε = 0.03 fixes everything.

---

## What we expected, and what we found

The original hypothesis was that a supervised model trained on confounded logs would **route badly**. It did not reproduce. Measured over 10 seeds, a well-regularised gradient-boosted model (`direct`) is a *strong, robust* policy on smooth latent structure — it has the **highest** true value of the four methods and its self-estimate of its own cautious policy looks honest.

The real failure is narrower and sharper, and it is **specific to starved regions**. `direct`'s model is accurate everywhere the legacy policy explored and **confidently wrong exactly where it didn't**: it predicts `pa`'s success on large tickets as ~0.78 when the truth is 0.68 (a persistent +0.10 gap), because the legacy policy sends everything above ₹5,000 to `pb` and `pa` is never observed there.

![pa predicted vs true success by ticket size](artifacts/extrapolation.png)

Because the phenomenon only bites in unexplored regions, the simulator's one hidden fact (`pa`'s −0.18 large-ticket weakness) was deliberately and transparently placed in the large-ticket region the legacy policy starves. That placement *is* the phenomenon, not a thumb on the scale — see `NOTES.md` (2026‑09‑04 entries "CRITICAL FINDING" and "Converged on the MINIMAL design") for the full, dated reasoning.

**Generality.** The finding does not depend on the specific legacy rule used here. Any non-random routing policy produces regions with near-zero coverage; the starved region moves, the problem does not. Razorpay Optimizer documents amount, method and issuer as routing conditions, which is why an amount-based rule was chosen.

---

## Results table

Four methods, learned from the same 200,000 confounded legacy logs (`switchyard` additionally gets a 200k epoch with 3% of decisions randomised). True value is by simulator rollout (10 seeds, common random numbers, 1000-resample paired bootstrap). All values are **₹ per 1,000 attempts**.

| Method | Estimated | True | Estimation error | True 95% CI | Improvement over legacy (CI) | Weights clipped |
|---|---:|---:|---:|:---:|:---:|---:|
| direct | 42410.0 | 41763.5 | +646.5 | [41436, 42128] | +1646 [1626, 1667] | 0 |
| ips | 42994.3 | 40436.1 | +2558.2 | [40127, 40775] | +319 [289, 347] | 0 |
| snips | 43070.3 | 40625.6 | +2444.7 | [40317, 40968] | +508.5 [481, 538] | 0 |
| dr | 42986.6 | 41005.4 | +1981.3 | [40686, 41357] | +888 [863, 914] | 0 |
| **switchyard** | 42180.4 | **41238.3** | +942.1 | [40918, 41597] | +1121 [1100, 1142] | 4039 |

Legacy baseline true value **40117.1**; oracle **42325.3**. A **fee-blind success-rate router** (route to highest predicted success, ignoring fees) scores **39861.9 — below legacy**: optimising success rate loses money; the objective is expected net revenue. Every learned method's improvement CI clears zero.

The off-policy value estimators all **overstate their own policy** (ips +2558, snips +2445, dr +1981 per 1k). `switchyard` overstates least of that family (+942) *and* beats them all on true value.

---

## Policy value: direct beats switchyard on raw value

Stated plainly, and not buried: **`direct` has the higher raw policy value.** On the held-out regime (a different traffic mix and degradation schedule, evaluated exactly once), the true-value ordering is:

| Method | True value, held-out (₹/1k) |
|---|---:|
| direct | 53516.1 |
| **switchyard** | 52776.2 |
| dr | 52285.1 |
| snips | 51424.7 |
| ips | 51253.8 |

(legacy 51312.3, oracle 54074.9.) The ordering is preserved from the main regime — the learned policies generalise, and switchyard beats every off-policy method — but `direct`'s smooth model is the higher-value *policy*. Switchyard's contribution is not a bigger policy; it is that its value estimates stay **honest** into the unexplored region (Headline), and `direct`'s do not — so an operator using `direct` cannot tell a good policy from a money-loser exactly where it matters.

### `hdfc × upi` segment picks (vs the true best)

| Ticket bucket | True best | direct | ips | snips | dr | switchyard |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| <₹300 | pb | ✓ | ✓ | ✓ | ✓ | ✓ |
| ₹300–1k | pb | ✓ | ✓ | ✓ | ✓ | ✓ |
| ₹1k–3k | pa | ✓ | ✓ | ✓ | ✓ | ✓ |
| ₹3k–10k | pa | ✓ | ✓ | ✓ | ✓ | ✓ |
| **>₹10k** | **pc** | pa ✗ | pa ✗ | ✓ | ✓ | pa ✗ |

On well-covered ticket sizes every method picks correctly. On the starved **>₹10k** bucket the truly-best `pc` is starved for *everyone* (≤46 samples even after exploration), so it stays contested — reported as measured, not smoothed.

### Ticket-size crossover (why fees matter)

![expected net reward vs ticket size](artifacts/crossover.png)

On `hdfc × upi`, `pa` has the highest success rate (0.95 vs pb 0.86, pc 0.85) yet is **not** the best expected-reward choice below **₹962** — its flat ₹4 fee eats the thin margin on small tickets.

---

## Money recovered by the recovery loop

When a payment fails, the same routing brain decides whether and where to re-attempt. Over **5,000 real failures**, switchyard's smart rerouting recovers **₹6,361.63 more** than the legacy retry policy — **127.23 paise/failure**, 95% CI **[113.22, 140.84]**, which excludes zero.

**Stopping rules** (enforced; live engine over the 5,000 failures): never re-attempt if expected net reward ≤ 0, at most 3 attempts, never re-attempt a hard decline. Outcomes: **3,489 recovered**, 1,447 hard declines correctly not retried, 51 hit the attempt cap, 13 stopped on non-positive expected reward. Every attempt is appended to `artifacts/audit.jsonl`.

**Idempotency:** a SQLite table with `txn_id` primary key + a `PENDING` reservation; the 10-concurrent-identical-events test yields exactly one attempt, and replaying a processed failure creates no second attempt.

**Compliance:** an attempt cap and a minimum inter-attempt delay for e-mandate-style (netbanking) failures are enforced. **RBI 24-hour pre-debit notification is NOT implemented** — a real gap a production system would need.

---

## Diagnosis component (LLM) and its scores

A single contained job (never routes): given a cohort's failure-code counts and a baseline comparison, output structured JSON `{cause, confidence, evidence}` or abstain with `INSUFFICIENT_EVIDENCE`. Output is schema-validated with a fallback to `INSUFFICIENT_EVIDENCE` on any malformed response, and cached by input hash. **Abstention is rewarded, not penalised**, because a calibrated "I don't know" is the correct answer for an un-diagnosable cohort and is strictly better than a confident wrong guess.

**Measured by a real model — Gemini `gemini-3.8-flash` (temperature 0, thinking disabled, strict JSON).** This was a **partial run**: the free tier's `PerDayPerProjectPerModel` quota is **20 requests/day**, which was exhausted after **9 of 19 cohorts** completed.

| Metric (Gemini, 9/19 cohorts completed) | Value |
|---|---:|
| Accuracy on the completed (all clear) cohorts | **1.000** |
| Parse-failure rate | 0.000 |
| Total tokens | 3,154 in / 847 out |

The 9 completed cohorts were all *clear* cohorts (they run first), so Gemini did **not** reach the ambiguous cohorts that test abstention. Those scores reproduce from the committed cache with no key; a rerun after the daily quota resets completes the rest.

**Abstention — measured by the deterministic offline statistical diagnoser, which is NOT a language model** (all 19 cohorts; reported here because Gemini's quota blocked the ambiguous cohorts):

| Metric (offline statistical, all 19 cohorts — NOT LLM) | Value |
|---|---:|
| Accuracy on clear cohorts | 0.923 |
| Abstention rate on ambiguous cohorts (rewarded) | 0.667 |
| Harmful-error rate (a wrong assertion) | 0.105 |

These offline numbers are **not** presented as language-model performance. The offline baseline abstains correctly on tiny cohorts but over-asserts on genuine two-cause blends (the 0.105 harmful rate) — exactly where an LLM's reasoning is expected to help.

**Disclosure:** provider Gemini, model `gemini-3.8-flash`; 3,154 input / 847 output tokens; parse-failure rate 0.000. Gemini free-tier inputs are used to improve Google's products; **all inputs are synthetic failure-code counts containing no real transaction or customer data.** OpenAI (`gpt-4o-mini`) is wired as a fallback for a non-rate-limit Gemini failure only, with a $2 spend tripwire; it was not invoked.

---

## Known limitations (real, unsmoothed)

1. **The naive model does not simply "fail."** A well-regularised GBM is a strong, robust *policy* on smooth latent structure — it beats the off-policy estimators on true value. The failure this project demonstrates is *epistemic and conditional on coverage*: `direct` cannot tell a good policy from a money-loser in the region it never explored, and overstates any policy that ventures there.
2. **ε = 0.03 buys honesty, not a bigger policy.** At this budget switchyard's edge is trustworthy value estimates, not a large true-value gain; the deepest-starved cells (e.g. `pc` on >₹10k, the adversarial trap) are not fully corrected — the price curve shows what a larger budget would cost.
3. **The simulator models each attempt as an independent draw** (no failure persistence), so absolute recovery counts (3,489/5,000) are optimistic; the *incremental-over-legacy* metric, which is headlined, is unaffected.
4. **Routing is over discretised `(method, issuer, amount-bucket)` cells**, not continuous amount — a deliberate choice so all four methods share one hypothesis space, at the cost of within-bucket resolution near crossovers.
5. **The soft degradation regime is unlearnable by every method** (no day-of-week feature), so no method routes around it — latent noise, not a demonstrated win.
6. **The diagnosis LLM run is partial** (9/19 cohorts) because of a 20-request/day free-tier quota; abstention is reported only from the offline baseline, clearly labelled as non-LLM.
7. **RBI 24-hour pre-debit notification is not implemented** (recovery §).

---

## One-command reproduction

```bash
pip install -r requirements.txt
python verify.py          # runs tests, regenerates every artifact, reprints every number above
```

`verify.py` exits non-zero on any failure and is deterministic on fixed seeds (byte-identical logs/tables, enforced by `tests/test_determinism.py`). The diagnosis numbers reproduce from the committed Gemini cache with **no API key**; set `GEMINI_API_KEY` (see `.env.example`) to complete the remaining cohorts after the daily quota resets.

---

## Prior art (cited by name)

The routing machinery here is established; **this project's contribution is the evaluation layer** — measuring when a model trained on logged routing decisions can and cannot be trusted, and pricing the exploration that fixes it.

- **Razorpay's published router** — Bygari, Gupta et al., *"An Intelligent Payment Routing System"*, IEEE Big Data 2021.
- **PayU** — payment success-rate routing, WWW 2018.
- **Juspay** — payment orchestration / smart routing (industry).
- **Adyen** — contextual-bandit payment routing, arXiv:2412.00569.
- Off-policy evaluation foundations: **inverse propensity weighting** (Horvitz & Thompson, 1952); **doubly robust** estimation (Robins, Rotnitzky & Zhao, 1994; Dudík, Langford & Li, 2011); **self-normalised IPS / counterfactual risk minimisation** (Swaminathan & Joachims, 2015); **counterfactual learning systems** (Bottou et al., 2013); **off-policy evaluation in contextual bandits** (Li, Chu, Langford & Schapire, 2010–11); the **optimizer's curse** (Smith & Winkler, 2006); **offline RL under distributional shift** (Levine, Kumar, Tucker & Fu, 2020).

See `DECISIONS.md` for design rationale and `NOTES.md` for the timestamped build log, including every simulator-tuning decision and the diagnosis quota event.
