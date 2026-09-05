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
