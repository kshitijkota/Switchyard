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

CAUSE_CLASSES = (CAUSE_ISSUER, CAUSE_MERCHANT, CAUSE_NETWORK, CAUSE_CUSTOMER)
VALID_OUTPUTS = CAUSE_CLASSES + (INSUFFICIENT,)

# Documented public meaning of each code (the code's label, known to any operator)
# and the cause family it belongs to. Not latent truth — a code's meaning is
# printed in the API docs.
CODE_MEANING = {
    "U30": ("bank declined", CAUSE_ISSUER),
    "U69": ("insufficient funds", CAUSE_CUSTOMER),
    "U16": ("risk rejected", CAUSE_CUSTOMER),
    "BAD_REQUEST_ERROR": ("malformed request", CAUSE_MERCHANT),
    "GATEWAY_ERROR": ("gateway error", CAUSE_NETWORK),
    "U67": ("timeout at PSP", CAUSE_NETWORK),
}
CODES = tuple(CODE_MEANING)

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
