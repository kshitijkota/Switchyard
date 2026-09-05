"""Latent ground truth — AGENT_BRIEF §4.2.

    ⚠️  THIS MODULE MUST NEVER BE IMPORTED FROM policy/ OR estimators/.  ⚠️

A test (tests/test_ground_truth_isolation.py) walks the AST of those packages
and fails if `sim.ground_truth` appears in any import. Do not weaken it.

It holds two latent structures the models are not allowed to see:
  1. the true success probability per (context, processor), and
  2. the true *cause class* behind each failure (only the observable failure
     CODE reaches the event log; the cause class is written to a separate
     ground-truth file, never into logs the models consume).
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime

import numpy as np

from events import Context

# --- Calendar -------------------------------------------------------------------
SIM_START = date(2026, 1, 1)


def day_index(ts: datetime) -> int:
    """Days since the simulation epoch (0-based)."""
    return (ts.date() - SIM_START).days


# --- Base success probabilities by processor × method (≈0.85–0.90) --------------
# Modest spread on purpose: if the base rates were far apart every method would
# find the winner and there would be no result to demonstrate (brief §4.2).
_BASE = {
    "pa": {"upi": 0.88, "card": 0.86, "netbanking": 0.85},
    "pb": {"upi": 0.86, "card": 0.88, "netbanking": 0.86},
    "pc": {"upi": 0.87, "card": 0.87, "netbanking": 0.88},
}

# Interaction thresholds (paise).
_TEN_K_PAISE = 1_000_000          # ₹10,000
_HIGH_AMOUNT_PAISE = 500_000      # ₹5,000 — legacy's high-amount cutoff (§5)
_PROB_FLOOR, _PROB_CEIL = 0.50, 0.98
_DEGRADE_DAYS = frozenset({2, 5})  # day_index % 7 in {2, 5}


def is_issuer_degraded(context: Context, processor: str) -> bool:
    """The soft issuer degradation: pa + sbi + upi, hours 14–18, on days where
    day_index % 7 ∈ {2, 5}. Lowers success AND skews failures to U30 (§4.2)."""
    return (
        processor == "pa"
        and context.issuer == "sbi"
        and context.method == "upi"
        and 14 <= context.hour <= 18
        and day_index(context.ts) % 7 in _DEGRADE_DAYS
    )


def success_prob(context: Context, processor: str) -> float:
    """True P(success | context, processor), clamped to [0.50, 0.98]."""
    p = _BASE[processor][context.method]
    if processor == "pa" and context.issuer == "hdfc" and context.method == "upi":
        p += 0.07                                   # §4.2 fact (well covered ⇒ learnable)
    if processor == "pb" and context.method == "card" and context.amount_paise > _TEN_K_PAISE:
        p += 0.04
    if processor == "pc":
        p -= 0.02                                   # pc anywhere
    if is_issuer_degraded(context, processor):
        p -= 0.15                                   # soft degradation
    # --- The one hidden fact placed in a legacy-STARVED region (v4, see NOTES) ---
    # Legacy sends >₹5k almost entirely to pb, so pa's large-ticket weakness is
    # never observed: a model trained on these logs cannot learn it.
    if processor == "pa" and context.amount_paise > _HIGH_AMOUNT_PAISE:
        p -= 0.18
    return float(min(_PROB_CEIL, max(_PROB_FLOOR, p)))


# --- Failure cause classes (the second latent structure) ------------------------
CAUSE_ISSUER = "ISSUER_DEGRADATION"
CAUSE_MERCHANT = "MERCHANT_INTEGRATION"
CAUSE_NETWORK = "NETWORK_TRANSIENT"
CAUSE_CUSTOMER = "CUSTOMER_SIDE"
CAUSE_CLASSES = (CAUSE_ISSUER, CAUSE_MERCHANT, CAUSE_NETWORK, CAUSE_CUSTOMER)

# Real published failure codes (TASK B). UPI path uses NPCI UPI response codes;
# card/netbanking uses Razorpay's documented error codes. Sources + retrieval
# dates are cited in DECISIONS.md. This code -> true-cause map is GROUND TRUTH and
# lives here only; the event log the models consume carries the bare code string,
# never the cause. `CAUSE_AMBIGUOUS` marks a code whose cause is genuinely not
# determinable from the code alone (its own published meaning names two causes).
CAUSE_AMBIGUOUS = "AMBIGUOUS"

CODE_TO_CAUSE = {
    # NPCI UPI response codes
    "U28": CAUSE_ISSUER,      # remitter/customer bank (PSP) is down
    "Z9":  CAUSE_CUSTOMER,    # insufficient funds in the customer's account
    "U69": CAUSE_CUSTOMER,    # collect request expired (customer took too long)
    "U30": CAUSE_AMBIGUOUS,   # "debit failed: bank down OR debit issue" — undiagnosable
    # Razorpay card / netbanking error codes
    "BAD_REQUEST_ERROR": CAUSE_MERCHANT,  # invalid request (integration/merchant)
    "GATEWAY_ERROR":     CAUSE_NETWORK,   # transient gateway/bank error, retryable
    "SERVER_ERROR":      CAUSE_NETWORK,   # transient internal error, retryable
}
# Codes each cause draws from when a failure is attributed to it. The ambiguous
# code U30 is NOT here — it is emitted from BOTH issuer and network failures (see
# sample_outcome), so a U30-dominated cohort cannot be told apart.
CAUSE_TO_CODES = {
    CAUSE_ISSUER:   ("U28",),
    CAUSE_CUSTOMER: ("Z9", "U69"),
    CAUSE_NETWORK:  ("GATEWAY_ERROR", "SERVER_ERROR"),
    CAUSE_MERCHANT: ("BAD_REQUEST_ERROR",),
}
AMBIGUOUS_CODE = "U30"
AMBIGUOUS_CAUSES = (CAUSE_ISSUER, CAUSE_NETWORK)
# Fraction of issuer/network failures that surface as the ambiguous U30 ("debit
# failed") instead of the clean cause-specific code. Chosen up front (NOT tuned to
# any result): large enough to build a genuinely-ambiguous U30 cohort, small
# enough that the clear issuer/network cohorts stay dominated by their own codes.
AMBIGUOUS_EMIT_PROB = 0.35

# Baseline mixture of failure causes (what the ambient failures look like).
_BASE_MIX = {CAUSE_CUSTOMER: 0.55, CAUSE_NETWORK: 0.25, CAUSE_ISSUER: 0.12, CAUSE_MERCHANT: 0.08}
# Regime mixtures — each concentrates one cause so the diagnoser has a signal to
# find against baseline. These shape *which* code a failure gets; they do not by
# themselves change whether a txn fails (except issuer degradation, which also
# lowers success_prob above).
_ISSUER_MIX = {CAUSE_ISSUER: 0.75, CAUSE_CUSTOMER: 0.15, CAUSE_NETWORK: 0.07, CAUSE_MERCHANT: 0.03}
_NETWORK_MIX = {CAUSE_NETWORK: 0.70, CAUSE_CUSTOMER: 0.20, CAUSE_ISSUER: 0.05, CAUSE_MERCHANT: 0.05}
_MERCHANT_MIX = {CAUSE_MERCHANT: 0.65, CAUSE_CUSTOMER: 0.22, CAUSE_NETWORK: 0.10, CAUSE_ISSUER: 0.03}


def network_incident(context: Context) -> bool:
    """Transient network window: the quiet small hours 2–3 see a NETWORK spike."""
    return context.hour in (2, 3)


def merchant_glitch(context: Context, processor: str) -> bool:
    """A specific merchant/integration pairing that emits BAD_REQUEST_ERRORs:
    netbanking on icici. Modest; affects only the failure-cause mixture."""
    return context.method == "netbanking" and context.issuer == "icici"


def regime(context: Context, processor: str) -> str:
    """The latent regime shaping this txn's failure-cause mixture. Precedence
    matters: issuer degradation dominates, then merchant glitch, then network."""
    if is_issuer_degraded(context, processor):
        return "issuer_degraded"
    if merchant_glitch(context, processor):
        return "merchant_glitch"
    if network_incident(context):
        return "network_incident"
    return "baseline"


_REGIME_MIX = {
    "issuer_degraded": _ISSUER_MIX,
    "merchant_glitch": _MERCHANT_MIX,
    "network_incident": _NETWORK_MIX,
    "baseline": _BASE_MIX,
}

# The cause class each non-baseline regime is engineered to concentrate. Used by
# the §8 diagnosis grader to label cohorts. 'baseline' has no dominant cause.
REGIME_TRUE_CAUSE = {
    "issuer_degraded": CAUSE_ISSUER,
    "merchant_glitch": CAUSE_MERCHANT,
    "network_incident": CAUSE_NETWORK,
}


def _cause_mix(context: Context, processor: str) -> dict:
    return _REGIME_MIX[regime(context, processor)]


def _ambiguous_draw(context: Context, processor: str) -> float:
    """A deterministic pseudo-random value in [0,1) for the U30 override, keyed on
    txn_id+processor. Crucially it consumes NO draws from the outcome RNG — the
    failure CODE is a cosmetic label and must never perturb the success/reward
    stream (see NOTES 2026-09-05). So the economic outcome is invariant to the
    failure-code taxonomy, and the routing experiments reproduce unchanged."""
    h = hashlib.sha256(f"{context.txn_id}|{processor}|u30".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def sample_outcome(context: Context, processor: str, rng: np.random.Generator):
    """Draw (success, failure_code, true_cause).

    On success -> (True, None, None). On failure -> (False, <code>, <cause>),
    where the cause is drawn from the regime mixture and the code from that
    cause's codes. Deterministic given `rng` state. The RNG draws (success, cause,
    code index) are IDENTICAL regardless of the code taxonomy, so success/reward
    outcomes do not depend on which failure codes exist.
    """
    p = success_prob(context, processor)
    if rng.random() < p:
        return True, None, None
    mix = _cause_mix(context, processor)
    causes = list(mix.keys())
    weights = np.array([mix[c] for c in causes], dtype=np.float64)
    weights /= weights.sum()
    cause = causes[int(rng.choice(len(causes), p=weights))]
    codes = CAUSE_TO_CODES[cause]
    code = codes[int(rng.integers(len(codes)))]           # always consumed (preserves stream)
    # Some issuer/network failures surface as the ambiguous U30 ("debit failed"),
    # whose published meaning names two causes — so the code alone cannot attribute
    # them. Chosen by a hash (no RNG draw), so the economic stream is unchanged; the
    # latent true_cause is still recorded (issuer or network), only the code is
    # ambiguous.
    if cause in AMBIGUOUS_CAUSES and _ambiguous_draw(context, processor) < AMBIGUOUS_EMIT_PROB:
        code = AMBIGUOUS_CODE
    return False, code, cause


# --- Vectorised success probability for eval rollouts ---------------------------

def success_prob_batch(
    methods: np.ndarray,
    issuers: np.ndarray,
    amounts: np.ndarray,
    hours: np.ndarray,
    day_indices: np.ndarray,
    processor: str,
    degrade_residues: tuple = (2, 5),
) -> np.ndarray:
    """Vectorised success_prob for a whole batch routed to one processor.

    With the default degrade_residues=(2,5) it returns exactly what
    success_prob() returns element-wise (a test checks this). The held-out regime
    passes a DIFFERENT degradation schedule.
    """
    n = len(methods)
    p = np.empty(n, dtype=np.float64)
    for m in ("upi", "card", "netbanking"):
        p[methods == m] = _BASE[processor][m]
    if processor == "pa":
        p[(issuers == "hdfc") & (methods == "upi")] += 0.07
        degr = (
            (issuers == "sbi") & (methods == "upi") & (hours >= 14) & (hours <= 18)
            & np.isin(np.mod(day_indices, 7), degrade_residues)
        )
        p[degr] -= 0.15
        p[amounts > _HIGH_AMOUNT_PAISE] -= 0.18                 # starved: pa on large
    elif processor == "pb":
        p[(methods == "card") & (amounts > _TEN_K_PAISE)] += 0.04
    elif processor == "pc":
        p -= 0.02
    return np.clip(p, _PROB_FLOOR, _PROB_CEIL)
