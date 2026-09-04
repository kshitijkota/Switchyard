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
  are right, and materially lower-variance than IPS — the right base for chowk.
- **Why weight clipping at 50.** chowk's data includes the ε-exploration slice
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
- **Why the LLM is confined to diagnosis** — _to be written with step 9._
