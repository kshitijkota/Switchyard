# Switchyard

**Switchyard routes each payment to the processor that maximises expected net revenue, and — unlike a supervised model trained on confounded logs — spends a small, budgeted slice of randomly-routed traffic so its value estimates stay honest in the segments the old policy never explored.**

> ## ⚠️ All data here is synthetic.
> Every number below is produced by a simulator in this repository with **known, controlled latent ground truth** — no real payment data is used, and no claim is made about any production system. No figure is hand-written; each is emitted by a script and reproduced end-to-end by `python verify.py`. *One thing is real:* the **failure codes and their documented meanings** are the published NPCI UPI and Razorpay error codes (see the diagnosis section and `DECISIONS.md`); the routing outcomes, success probabilities and costs attached to them are simulated.

---

## Money recovered by the recovery loop

When a payment fails, Switchyard's routing brain decides whether and where to re-attempt. Over **5,000 simulated payment failures**, switchyard's smart rerouting recovers **₹6,361.63 more** than the legacy retry policy — **127.23 paise/failure**, 95% CI **[113.22, 140.84]**, which excludes zero.

**Stopping rules** (enforced; live engine over the 5,000 failures): never re-attempt if expected net reward ≤ 0, at most 3 attempts, never re-attempt a hard decline. Outcomes: **3,489 recovered**, 1,447 hard declines correctly not retried, 51 hit the attempt cap, 13 stopped on non-positive expected reward. Every attempt is appended to `artifacts/audit.jsonl`.

**Idempotency:** a SQLite table with `txn_id` primary key + a `PENDING` reservation; the 10-concurrent-identical-events test yields exactly one attempt, and replaying a processed failure creates no second attempt.

**Compliance:** an attempt cap and a minimum inter-attempt delay for e-mandate-style (netbanking) failures are enforced. **RBI 24-hour pre-debit notification is NOT implemented** — a real gap a production system would need.

---

## Where semantics actually matter: an open-world test (rule vs trained model vs LLM)

The closed-world result (a rule at 0.923 matching gpt-4o-mini at 0.846), reported in the diagnosis section below, is real but says little — seven documented codes and five fixed categories are a lookup, not an interpretation. So we add a **held-out open-world set** the rule cannot be pre-written for, and a **third method** — a small trained classifier (multinomial logistic regression on documented-code shares) — that sits between the rule and the LLM. The open-world cohorts (labels fixed before any method ran) contain: **real NPCI/Razorpay codes withheld** from the table and the classifier's training (`Z8`, `U16`, `bank_not_available`, `psp_app_not_available`, `gateway_technical_error`); **free-text gateway messages** with no code; a **red herring** (the ambiguous `U30` dominates, the true signal is a minority code); and **two-cause blends**.

**Prediction, stated before the results:** rules and trained models should **win closed-world and fail open-world** (they can only match patterns they were given); the LLM should **lose closed-world and generalise open-world** (it can interpret an unfamiliar code or a free-text message from world knowledge).

**What actually happened** — accuracy on clear cohorts (harmful-error rate in parentheses):

| Method | Closed-world | Open-world |
|---|---:|---:|
| Hand-written rule | 0.923 (0.10) | **0.000** (0.10) |
| Trained classifier (LR on code shares) | **1.000** (0.00) | 0.125 (0.40) |
| LLM (gpt-4o-mini) | 0.846 (0.10) | **0.625** (0.30) |

**The prediction held.** Closed-world, the trained classifier is perfect (1.000) and the rule strong (0.923) — both beat the LLM (0.846). Open-world, both collapse and the **LLM is the only method that generalises** (0.625 vs 0.000 / 0.125): it reads `bank_not_available`, `Z8` and every free-text cohort correctly from knowledge it was never given in a table. And the two failures are different in a way that mirrors this whole project: the **rule fails safe** — it abstains (0.0 accuracy, but only 0.10 harmful), correctly refusing to interpret codes it doesn't recognise; the **trained model fails dangerously** — it extrapolates its closed-world mapping onto inputs it never saw and asserts a **confident wrong cause 40% of the time** (0.40 harmful), the same optimism-under-distribution-shift that the routing half of this project is about. The LLM is not clean either (0.30 harmful — it misreads `psp_app_not_available` and `gateway_technical_error` as customer-side, and over-asserts on one blend), and it does **not** win closed-world.

**Abstention attribution.** The diagnoser's headline abstention is mostly deterministic engineering, not model judgment. On the closed-world ambiguous cohorts the model abstains **0.714 including the sample-size guardrail**, but **4 of those 5 abstentions are the guardrail** — a hard "abstain below 40 failures, no model call" gate. The model's **own** abstention, on the cohorts that actually reached it, is **0.188** (3 of 16). The guardrail was added after three prompt variants failed to move the number (the model kept asserting on tiny cohorts, abstention stuck at 0.167) — deterministic engineering, not model behaviour.

*Reproduce:* `python -m diagnose.threeway` (rule and trained classifier need no key; the LLM reproduces from the committed OpenAI cache). All open-world inputs are synthetic; only the withheld codes and their meanings are real.

---

## Headline

The routing half of the project measures the same distribution-shift failure in rupees.

Point each estimator at a routing policy that relies on the region the legacy policy never explored, and ask what it's worth — the truth is known by simulation.

- On an **adversarial policy that routes _all_ large tickets to `pa`** — a genuine money-loser, true value **₹38,974 / 1k** attempts, *below* the ₹40,117 legacy baseline — **`direct` values it ₹1,954/1k too high.** It would green-light the loser with no warning signal.
- On **`direct`'s own deployed greedy policy** (true value ₹41,763/1k), **`switchyard` is off by just ₹17.3/1k** — while `direct` overstates by ₹646.5 and plain IPS is off by ₹2,100.7. Switchyard has the lowest estimation error among the tested estimators in this unexplored region.

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

**Where the curve stops being worth it:** the estimation error on the starved policy crosses honest (~0) at **ε ≈ 0.03**, then *over-corrects* to −₹254 at ε = 0.10 while the exploration cost more than triples (₹47 → ₹157/1k). So the knee is ε ≈ 0.03: near-zero error at modest cost, and nothing bought beyond it. The adversarial policy is starved so deeply (it forces *every* large ticket onto the unexplored `pa`) that even ε = 0.10 only trims its error from ₹1,609 to ₹1,265/1k — evaluating it honestly would need a larger exploration budget than ε = 0.03.

---

## What we expected, and what we found

The original hypothesis was that a supervised model trained on confounded logs would **route badly**. It did not reproduce. Measured over 10 seeds, a well-regularised gradient-boosted model (`direct`) is a *strong, robust* policy on smooth latent structure — it has the **highest** true value of the four methods and its self-estimate of its own cautious policy looks honest.

The real failure is narrower and sharper, and it is **specific to starved regions**. `direct`'s model is accurate everywhere the legacy policy explored and **confidently wrong exactly where it didn't**: it predicts `pa`'s success on large tickets as ~0.78 when the truth is 0.68 (a persistent +0.10 gap), because the legacy policy sends everything above ₹5,000 to `pb` and `pa` is never observed there.

![pa predicted vs true success by ticket size](artifacts/extrapolation.png)

Because the phenomenon only bites in unexplored regions, the simulator's one hidden fact (`pa`'s −0.18 large-ticket weakness) sits in the large-ticket region the legacy policy starves — that placement is the phenomenon itself. See `NOTES.md` (2026‑09‑04 entries "CRITICAL FINDING" and "Converged on the MINIMAL design") for the full reasoning.

**Generality.** The finding does not depend on the specific legacy rule used here. Any non-random routing policy produces regions with near-zero coverage; the starved region moves, the problem does not. Razorpay Optimizer documents amount, method and issuer as routing conditions, which is why an amount-based rule was chosen.

---

## Results table

Methods learned from the same 200,000 confounded legacy logs (`switchyard` additionally gets a 200k epoch with 3% of decisions randomised). `bygari_baseline` is a faithful reimplementation of the routing architecture described in Bygari et al. (2021) that routes on predicted **success**. True value is by simulator rollout (10 seeds, common random numbers, 1000-resample paired bootstrap). All values are **₹ per 1,000 attempts**.

| Method | Estimated | True | Estimation error | True 95% CI | Improvement over legacy (CI) | Weights clipped |
|---|---:|---:|---:|:---:|:---:|---:|
| direct | 42410.0 | 41763.5 | +646.5 | [41436, 42128] | +1646 [1626, 1667] | 0 |
| ips | 42994.3 | 40436.1 | +2558.2 | [40127, 40775] | +319 [289, 347] | 0 |
| snips | 43070.3 | 40625.6 | +2444.7 | [40317, 40968] | +508.5 [481, 538] | 0 |
| dr | 42986.6 | 41005.4 | +1981.3 | [40686, 41357] | +888 [863, 914] | 0 |
| **switchyard** | 42180.4 | **41238.3** | +942.1 | [40918, 41597] | +1121 [1100, 1142] | 4039 |
| bygari_baseline (success-router) | 40208.5 | 40058.9 | +149.6 | [39740, 40419] | **−58 [−74, −42]** | 0 |

Legacy baseline true value **40117.1**; oracle **42325.3**. Two baselines below legacy: a **fee-blind success-rate router** scores **39861.9**, and **`bygari_baseline`**'s success-routing policy scores **40058.9 — below legacy and ₹1,705/1k below `direct`'s reward-routing.** Optimising success rate rather than expected net revenue loses money. Every *reward*-optimising method's improvement CI clears zero.

The off-policy value estimators all **overstate their own policy** (ips +2558, snips +2445, dr +1981 per 1k). `switchyard` overstates least of that family (+942) *and* beats them all on true value. **Offline, `switchyard`'s policy dominates `bygari_baseline` by ₹1,179/1k** — but the live online race below tells a different story.

---

## Live continuous operation: bygari_baseline vs switchyard — indistinguishable over 50k, switchyard ahead only over 2M

Run both as online learners in parallel, each starting from the same confounded legacy log and updating as outcomes arrive. `bygari_baseline` uses its published feedback loop (rolling time-decayed success rates + LR downtime breaker); the online `switchyard` is a per-cell ε-greedy net-revenue bandit. All figures below are over **40 seeds** with common random numbers (seeds 2000–2039).

![cumulative net revenue](artifacts/cumulative_value.png)

**H1 — exploration sweep (40 seeds × 50,000 txns/seed).** Final net revenue ₹/seed, `switchyard − bygari` by 1000-resample paired bootstrap:

| ε | bygari | switchyard | switchyard − bygari | 95% CI | verdict | crossover | exploration cost/seed |
|---:|---:|---:|---:|:---:|:---|:---:|---:|
| 0.01 | 2,009,354.9 | 2,009,676.8 | **+321.9** | [−1,126.2, +1,731.8] | indistinguishable | 44,000 txns | ₹371.7 |
| 0.03 | 2,009,354.9 | 2,008,262.2 | −1,092.7 | [−2,509.8, +465.0] | indistinguishable | none | ₹1,086.7 |
| 0.10 | 2,009,354.9 | 2,006,741.0 | **−2,613.9** | [−4,134.6, −1,041.5] | **distinguishable — switchyard loses** | none | ₹4,083.8 |

Over a 50k-txn window the two are **statistically indistinguishable at ε = 0.01 and 0.03** (CI includes zero), and at **ε = 0.10 `switchyard` is significantly _behind_** — the ₹4,084/seed exploration bill is not repaid inside 50k. More exploration hurts here, not helps. The best ε for switchyard is the cheapest one, **0.01**.

**H2 — horizon (best ε = 0.01, 2,000,000 txns/seed × 10 seeds = 20M txns).** Extending the same race 40× longer:

| horizon | bygari | switchyard | switchyard − bygari | 95% CI | verdict | crossover |
|---|---:|---:|---:|:---:|:---|:---:|
| 2M txns/seed | 80,318,802.8 | 80,696,013.6 | **+377,210.8** | [+236,094.7, +511,085.3] | **distinguishable — switchyard wins** | 160,000 txns |

Given enough volume, `switchyard`'s online learning **does** overtake the strong pretrained router and pull clearly ahead (+₹377k/seed, CI excludes zero). But it takes **~160,000 transactions** to cross over — far beyond the 50k window where the two look tied. The headline is therefore horizon-dependent and stated as such: **tied over 50k, switchyard ahead over 2M.**

**H3 — where the difference lives (starved large-ticket region vs remainder).** Large-ticket (>₹5,000) traffic is **12.7%** of the stream. Splitting the net difference by region (₹/seed at 50k):

| region (share) | switchyard − bygari | 95% CI | verdict |
|---|---:|:---:|:---|
| large-ticket >₹5k (12.7%), ε=0.03 | **−12,184.0** | [−13,576.9, −10,667.5] | distinguishable — **bygari wins the starved region** |
| small-ticket remainder (87.3%), ε=0.03 | **+11,091.3** | [+10,830.3, +11,362.2] | distinguishable — **switchyard wins the bulk** |

![starved-region traffic](artifacts/starved_region_traffic.png)

The expected pattern — that `bygari` would send ~0 traffic to the starved region and lose there — does **not** hold: `bygari` routes *more* large-ticket traffic to pa/pc (share **0.216**) than the online `switchyard` (share **0.108** at ε=0.03), and it **earns more in that region, not less.** `pc` is the *truly-best* large-ticket processor (cheapest per-paise on big amounts), but it is starved in switchyard's legacy initialisation, so the conservative online bandit stays anchored to `pb` and **under-routes to `pc`, losing ~₹12k/seed in the large-ticket region.** What switchyard wins is the **small-ticket bulk** (+₹11k/seed) — enough to offset the large-ticket loss to a wash over 50k, and to dominate over 2M (where the large-ticket gap itself washes out to indistinguishable, CI [−256k, +23k]).

**Bottom line:** against a strong pretrained success-router, this simple online `switchyard` is **not a free win** — indistinguishable over 50k, strictly worse at high ε, worse in the starved region, and decisively ahead only once given 2M transactions.

---

## Policy value: direct beats switchyard on raw value

**`direct` has the higher raw policy value.** On the held-out regime (a different traffic mix and degradation schedule, evaluated exactly once), the true-value ordering is:

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

On well-covered ticket sizes every method picks correctly. On the starved **>₹10k** bucket the truly-best `pc` is starved for *everyone* (≤46 samples even after exploration), so it stays contested and no reward-optimising method finds it.

### Ticket-size crossover (why fees matter)

![expected net reward vs ticket size](artifacts/crossover.png)

On `hdfc × upi`, `pa` has the highest success rate (0.95 vs pb 0.86, pc 0.85) yet is **not** the best expected-reward choice below **₹962** — its flat ₹4 fee eats the thin margin on small tickets.

---

## Diagnosis component (LLM) and its scores

A single contained job (never routes): given a cohort's failure-code counts and a baseline comparison, output structured JSON `{cause, confidence, evidence}` or abstain with `INSUFFICIENT_EVIDENCE`. Output is schema-validated with a fallback to `INSUFFICIENT_EVIDENCE` on any malformed response, and cached by input hash. A **sample-size guardrail** abstains deterministically (no model call) on cohorts below 40 failures — a production diagnoser must not attribute a cause from a handful of events. **Abstention is rewarded, not penalised**, because a calibrated "I don't know" is the correct answer for an un-diagnosable cohort and is strictly better than a confident wrong guess.

**Real failure codes (what is real vs synthetic).** The failure codes and their documented meanings are **real**: the UPI path uses NPCI's published UPI response codes (`U28` bank/PSP down, `Z9` insufficient funds, `U69` collect-expired, `U30` "debit failed — bank down **or** debit issue"), the card/netbanking path uses Razorpay's documented error codes (`BAD_REQUEST_ERROR`, `GATEWAY_ERROR`, `SERVER_ERROR`); sources and retrieval dates are in `DECISIONS.md`. Everything else — routing outcomes, success probabilities and costs — remains **simulated** with known latent ground truth. `U30` is **genuinely ambiguous** (its own published text names two causes), emitted from both issuer and network failures, so a U30-dominated cohort is un-diagnosable and the correct answer is `INSUFFICIENT_EVIDENCE`.

**Measured by a real model — OpenAI `gpt-4o-mini` (temperature 0, strict JSON), all 20 cohorts complete.**

| Metric (all 20 cohorts) | gpt-4o-mini | offline rule (not an LLM) |
|---|---:|---:|
| Accuracy on clear cohorts (13) | 0.846 (11/13) | **0.923** (12/13) |
| Abstention on ambiguous cohorts (7, rewarded) | 0.714 (5/7) | 0.714 (5/7) |
| Harmful-error rate (a wrong assertion) | 0.100 (2/20) | 0.100 (2/20) |
| Parse-failure rate | 0.000 | 0.000 |

**Both correctly abstain on the genuinely-ambiguous `U30` cohort** — the code names two causes, and neither the model nor the rule invents one. That is the headline of the real-code change, and it is the point of the ambiguity: a U30-dominated window is un-diagnosable, and both the model and the rule say so.

Beyond that, a hand-written rule (0.923) still slightly edges gpt-4o-mini (0.846) on clear-accuracy, with **identical** abstention (0.714) and harmful rates (0.100). The two make the *same* two harmful errors — the constructed two-cause blends, where each confidently names the larger cause instead of abstaining. The only difference is on the customer-baseline cohorts (a baseline window looks like normal ambient traffic, so "nothing anomalous" is a defensible read): the rule abstains on one of them, gpt-4o-mini on two, which separates 0.923 from 0.846. The offline rule is not a language model. That a small LLM does not beat a fixed seven-code lookup here — even with real codes — is what motivates the open-world test above.

**Disclosure:** provider OpenAI, model `gpt-4o-mini`; 9,779 input / 1,141 output tokens; **estimated cost $0.0022** (hard cap $20, $5 runaway tripwire — never approached); parse-failure rate 0.000. **All inputs are synthetic failure-code counts containing no real transaction or customer data.** The Gemini and Anthropic paths stay selectable; the offline provider runs when no key is present. Numbers reproduce from the committed OpenAI cache with no key.

---

## Known limitations

1. **The naive model does not simply "fail."** A well-regularised GBM is a strong, robust *policy* on smooth latent structure — it beats the off-policy estimators on true value. The failure this project demonstrates is *epistemic and conditional on coverage*: `direct` cannot tell a good policy from a money-loser in the region it never explored, and overstates any policy that ventures there.
2. **ε = 0.03 buys honesty, not a bigger policy.** At this budget switchyard's edge is trustworthy value estimates, not a large true-value gain; the deepest-starved cells (e.g. `pc` on >₹10k, the adversarial trap) are not fully corrected — the price curve shows what a larger budget would cost.
3. **The simulator models each attempt as an independent draw** (no failure persistence), so absolute recovery counts (3,489/5,000) are optimistic; the *incremental-over-legacy* metric, which is headlined, is unaffected.
4. **Routing is over discretised `(method, issuer, amount-bucket)` cells**, not continuous amount — a deliberate choice so all four methods share one hypothesis space, at the cost of within-bucket resolution near crossovers.
5. **The soft degradation regime is unlearnable by every method** (no day-of-week feature), so no method routes around it — latent noise, not a demonstrated win.
6. **On the closed-world, real-code diagnosis task a hand-written rule still slightly beats the small LLM** (rule 0.923 vs gpt-4o-mini 0.846 clear-accuracy; identical 0.714 abstention and 0.100 harmful — the *same* two blend errors): both correctly abstain on the genuinely-ambiguous `U30` code, both over-assert on the two-cause blends, and the gap is only that gpt-4o-mini abstains on one extra baseline window. A fixed seven-code lookup is simple enough that a small LLM does not beat it here — even with real codes — which is what the open-world test is for.
7. **The online `switchyard` in the live race is a simple per-cell ε-greedy bandit**, deliberately weaker than the batch DR estimator (no cross-cell generalisation, anchored to a heavy legacy prior). Over the 50k-transaction window it is statistically indistinguishable from `bygari_baseline` (at ε = 0.01 and 0.03 the difference CI includes zero); at 2,000,000 transactions/seed it pulls clearly ahead (+₹377,210.8/seed, CI excluding zero), with the crossover at ≈160,000 transactions. A model-based online learner might close the short-horizon gap, but that was not the committed design and is not claimed.
8. **RBI 24-hour pre-debit notification is not implemented** (recovery §).

---

## One-command reproduction

```bash
pip install -r requirements.txt
python verify.py          # runs tests, regenerates every artifact, reprints every number above (~15 min: includes the 40-seed live race and the open-world diagnosis eval)
```

`verify.py` exits non-zero on any failure and is deterministic on fixed seeds (byte-identical logs/tables, enforced by `tests/test_determinism.py`). The diagnosis numbers reproduce from the committed OpenAI `gpt-4o-mini` cache with **no API key**; set `OPENAI_API_KEY` or `GEMINI_API_KEY` (see `.env.example`) to re-run the model live.

---

## Prior art (cited by name)

The routing machinery here is established; **this project's contribution is the evaluation layer** — measuring when a model trained on logged routing decisions can and cannot be trusted, and pricing the exploration that fixes it.

- **Routing architecture** — the design reimplemented here as `bygari_baseline` (static eligibility + LR downtime breaker; dynamic RF success-router with time-decay feedback), run head-to-head above, follows the routing architecture described in Bygari et al. (2021): Ramya Bygari, Aayush Gupta, Shashwat Raghuvanshi, Aakanksha Bapna and Birendra Sahu, *"An AI-powered Smart Routing Solution for Payment Systems,"* 2021 IEEE International Conference on Big Data (Big Data); the authors are affiliated with Razorpay, Bengaluru.
- **Failure codes** — NPCI *UPI Error and Response Codes* (v2.9) and Razorpay's documented payment error codes are used verbatim for the failure taxonomy (see `DECISIONS.md` for URLs and retrieval dates); only the codes and meanings are real, the outcomes remain simulated.
- **PayU** — payment success-rate routing, WWW 2018.
- **Juspay** — payment orchestration / smart routing (industry).
- **Adyen** — contextual-bandit payment routing, arXiv:2412.00569.
- Off-policy evaluation foundations: **inverse propensity weighting** (Horvitz & Thompson, 1952); **doubly robust** estimation (Robins, Rotnitzky & Zhao, 1994; Dudík, Langford & Li, 2011); **self-normalised IPS / counterfactual risk minimisation** (Swaminathan & Joachims, 2015); **counterfactual learning systems** (Bottou et al., 2013); **off-policy evaluation in contextual bandits** (Li, Chu, Langford & Schapire, 2010–11); the **optimizer's curse** (Smith & Winkler, 2006); **offline RL under distributional shift** (Levine, Kumar, Tucker & Fu, 2020).

See `DECISIONS.md` for design rationale and `NOTES.md` for the timestamped build log, including every simulator-tuning decision and the diagnosis quota event.
