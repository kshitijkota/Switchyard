# DECISIONS

Design rationale. Expanded as each component lands (AGENT_BRIEF §11).

- **Why DR over plain IPS.** Plain IPS reweights logged rewards by 1/propensity;
  in the cells the legacy policy starved, propensities are tiny and sample counts
  small, so IPS has enormous variance — its per-cell argmax chases noise
  (optimiser's curse) and it overstates its own value by ₹2,000–3,000 per 1k txns
  in our runs. SNIPS (self-normalised) cuts the variance by dividing by the sum
  of weights, at the cost of a little bias. DR goes further: it starts from the
  model's prediction and only adds an IPW correction on the *residual*, so where
  there is no coverage the correction is zero and DR falls back to the model
  instead of exploding. DR is unbiased if EITHER the model or the propensities
  are right, and materially lower-variance than IPS — the right base for switchyard.
- **Why weight clipping at 50.** switchyard's data includes the ε-exploration slice
  whose minimum propensity is 0.01, i.e. importance weights up to 100. A handful
  of such samples would otherwise dominate a cell's estimate. Clipping weights at
  50 caps any single sample's leverage (trading a small downward bias for a large
  variance reduction); the number of clipped samples is reported so the bias is
  visible, not hidden.
- **Why SQLite WAL over an application mutex.** Idempotency must survive process
  restarts and concurrent workers, which an in-process mutex cannot: a mutex is
  lost on crash and does not coordinate across processes. A SQLite table with
  `txn_id` as PRIMARY KEY makes "has this txn been reserved?" a durable, atomic
  question — the first INSERT wins, concurrent duplicates hit the constraint and
  are turned away. WAL mode lets readers and a writer proceed without blocking
  each other and, with a busy timeout, serialises concurrent writers cleanly, so
  the 10-concurrent-identical-events test yields exactly one attempt. The store
  is the source of truth; no application-level lock is needed.
- **bygari_baseline constants (TASK A).** Faithful reimplementation of Bygari et
  al. (IEEE Big Data 2021). Eligibility table: all three processors are
  contractually eligible for every txn in this sim (no artificial restriction —
  the interesting behaviour is in the dynamic module, not a static allow-list).
  Adaptive time-decay feedback uses EWMA with `SUCCESS_DECAY=0.99` (rolling
  success rate) and `ERRVEL_DECAY=0.98` (error velocity); the LR downtime breaker
  trips at predicted failure prob > 0.60. RF: 60 trees, depth 14. These were
  chosen up front (typical EWMA half-lives ~70/35 events) and NOT tuned to the
  comparison outcome. Online routing freezes the rolling features within 100-txn
  mini-batches for tractability (per-txn RF predict would take ~2h for 500k);
  this if anything slows bygari's feedback, it does not favour switchyard.
- **Online switchyard design (TASK A).** A deliberately simple online analog: a
  per-(cell, processor) running mean of net reward initialised from the legacy
  log, ε-greedy (ε=0.03), updated online. It is weaker than the batch DR
  switchyard (no cross-cell model generalisation) — a fair, faithful online
  contextual bandit, not the batch estimator. This choice was committed before
  the run; the batch/online gap is itself part of the reported finding.
- **Why the LLM is confined to diagnosis.** Routing decides where real money
  goes, per transaction, at scale — it must be auditable, reproducible, and
  bounded, which the estimator/policy stack is and a free-text model is not. The
  LLM is given one contained, non-actuating job: read a cohort's failure-code
  distribution and name a likely cause (or abstain). It never routes, its output
  is schema-validated with an INSUFFICIENT_EVIDENCE fallback on any malformed
  response, and its answers are cached by input hash so evaluation is
  reproducible and cheap. Abstention is rewarded in scoring, so the model is
  never pushed to guess. This keeps the LLM's failure modes off the money path.
  (In this build environment no API key was available, so the reported numbers
  come from the deterministic offline statistical diagnoser; the LLM path runs
  and is scored identically when ANTHROPIC_API_KEY is set.)
- **Real published failure codes (TASK B).** The failure taxonomy is no longer
  invented. The UPI path uses NPCI's published UPI response codes; the
  card/netbanking path uses Razorpay's documented payment error codes. Only the
  failure codes and their documented meanings are real — routing outcomes,
  success probabilities and costs remain simulated with known latent ground
  truth. Codes and meanings used:
  - `U28` — remitter/customer bank (PSP) unavailable → ISSUER_DEGRADATION
  - `Z9` — insufficient funds in the customer's account → CUSTOMER_SIDE
  - `U69` — collect request expired, customer did not approve in time → CUSTOMER_SIDE
  - `U30` — "debit has failed … the customer's bank is down **or** there is an
    issue debiting the account" → **genuinely AMBIGUOUS** (issuer outage vs a
    technical debit failure). Its own published description names two causes, so
    it identifies none; a cohort dominated by U30 is un-diagnosable and the
    correct diagnosis is INSUFFICIENT_EVIDENCE.
  - `BAD_REQUEST_ERROR` — invalid request / integration error (source=business) → MERCHANT_INTEGRATION
  - `GATEWAY_ERROR` — transient gateway/bank error, retryable (source=gateway) → NETWORK_TRANSIENT
  - `SERVER_ERROR` — transient internal error, retryable → NETWORK_TRANSIENT
  The code→true-cause map lives in `sim/ground_truth.py` only; the event log the
  models consume carries the bare code string. U30 is emitted from **both**
  issuer and network failures (`AMBIGUOUS_EMIT_PROB=0.35`, chosen up front, not
  tuned) so a U30-dominated cohort genuinely mixes two causes. The recovery
  engine's non-retryable "hard decline" is now `Z9` (insufficient funds — routing
  the same payment elsewhere cannot create funds), replacing the old `U16`.
  Sources (retrieved 2026-09-05):
  - NPCI UPI codes cross-checked against Razorpay's guide "Tackling UPI Payment
    Failures" — https://razorpay.com/blog/tackling-upi-payment-failures-with-razorpay/
    (verbatim: U69 "collect request expired as the customer took more time to
    complete the payment"; Z9 "insufficient funds"; U28 "customer's bank … is
    down"; U30 "debit has failed. This can happen if the customer's bank is down
    or there is any issue in debiting the bank account") — and the canonical NPCI
    "UPI Error and Response Codes v2.9"
    (https://dth95m2xtyv8v.cloudfront.net/tesseract/assets/upi-tpap-sdk/UPI_Error_and_Response_Codes_2_9-HHLrJ.pdf).
  - Razorpay payment error codes, sources and reasons —
    https://razorpay.com/docs/errors/payments/list/ and
    https://razorpay.com/docs/errors/ (source field values customer / business /
    gateway / razorpay; reasons e.g. payment_failed, insufficient_funds,
    bank_not_available, gateway_technical_error, payment_method_not_enabled).
- **Open-world diagnosis test + trained classifier (TASK C).** The closed-world
  diagnosis (six documented codes, five categories) is a lookup, so a rule ties a
  small LLM there and the comparison is uninformative. TASK C adds (a) a held-out
  open-world cohort set the rule cannot be pre-written for — real but WITHHELD
  NPCI/Razorpay codes, free-text gateway messages, a minority-signal red herring,
  and two-cause blends, all labelled before any method ran — and (b) a third
  method, a multinomial logistic regression on documented-code shares, that sits
  between the rule and the LLM. Design choices, fixed up front:
  - Classifier features are the shares of the SEVEN documented codes only. By
    construction it has no representation for withheld codes or free-text — it is
    the closed-world view, so it should (and does) fail open-world. Trained on
    resampled labelled windows from the deterministic log, so it reproduces with
    no key; `ABSTAIN_THRESH=0.45` and an INSUFFICIENT class let it abstain.
  - The lookup rule abstains when fewer than 40 cohort failures carry a
    documented code (it must not attribute from the handful of ambient known codes
    in a withheld-code cohort) — the honest lookup behaviour.
  - Open-world interpretation guidance (untabled codes / free-text / "read the
    minority signal") is added to the LLM prompt ONLY for cohorts that actually
    carry unknown codes or free-text; closed-world cohorts get the unchanged base
    prompt. This scopes the guidance to where it is relevant and keeps the
    closed-world LLM number (0.846) identical to TASK B — the guidance is not a
    tuning knob applied to closed-world results. See NOTES for why (a first pass
    that put it in the base prompt made the model over-cautious on closed-world
    baseline cohorts).
  - The prediction (rules/trained win closed and fail open; LLM loses closed and
    generalises open) is stated in the README BEFORE the results table, per the
    governing rule, and the full grid — including the LLM's open-world errors — is
    reported as measured.
