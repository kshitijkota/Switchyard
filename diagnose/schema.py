"""Diagnosis output vocabulary and schema — AGENT_BRIEF §8.

The diagnoser's OUTPUT space and the PUBLIC documented meaning of each failure
code (§3.1's left column — what the code means, which any operator knows). This
is deliberately separate from sim.ground_truth: the diagnoser never learns which
cohort is *truly* in which regime — it infers that from the code distributions.
The node never imports ground truth.
"""

from __future__ import annotations

CAUSE_ISSUER = "ISSUER_DEGRADATION"
CAUSE_MERCHANT = "MERCHANT_INTEGRATION"
CAUSE_NETWORK = "NETWORK_TRANSIENT"
CAUSE_CUSTOMER = "CUSTOMER_SIDE"
INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
# A code whose published meaning names more than one cause (e.g. NPCI U30, "debit
# failed: bank down OR debit issue"). It is NOT a valid diagnosis output — the
# correct answer on a cohort dominated by such codes is INSUFFICIENT_EVIDENCE.
CAUSE_AMBIGUOUS = "AMBIGUOUS"

CAUSE_CLASSES = (CAUSE_ISSUER, CAUSE_MERCHANT, CAUSE_NETWORK, CAUSE_CUSTOMER)
VALID_OUTPUTS = CAUSE_CLASSES + (INSUFFICIENT,)

# Documented PUBLIC meaning of each code (its label as printed in the NPCI /
# Razorpay docs — TASK B) and the cause family that meaning implies. This is not
# latent truth; it is what any operator reads off the code. Codes marked
# CAUSE_AMBIGUOUS name two causes in their own documentation and so identify none.
CODE_MEANING = {
    # NPCI UPI response codes
    "U28": ("remitter/customer bank (PSP) unavailable", CAUSE_ISSUER),
    "Z9":  ("insufficient funds in the customer's account", CAUSE_CUSTOMER),
    "U69": ("collect request expired — customer did not approve in time", CAUSE_CUSTOMER),
    "U30": ("debit failed — customer's bank is down OR a technical debit failure", CAUSE_AMBIGUOUS),
    # Razorpay card / netbanking error codes
    "BAD_REQUEST_ERROR": ("invalid request — integration/merchant error", CAUSE_MERCHANT),
    "GATEWAY_ERROR":     ("payment gateway/bank transient error (retryable)", CAUSE_NETWORK),
    "SERVER_ERROR":      ("internal transient error at gateway/Razorpay (retryable)", CAUSE_NETWORK),
}
CODES = tuple(CODE_MEANING)
AMBIGUOUS_CODES = tuple(c for c, (_m, cause) in CODE_MEANING.items() if cause == CAUSE_AMBIGUOUS)

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cause", "confidence", "evidence"],
    "properties": {
        "cause": {"type": "string", "enum": list(VALID_OUTPUTS)},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    },
}
