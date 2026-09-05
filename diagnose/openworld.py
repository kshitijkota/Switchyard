"""Open-world diagnosis scenarios — TASK C1 (GRADER side; ground truth lives here).

The closed-world test uses six documented codes and five fixed categories — a
lookup table wins because there is nothing to interpret. These held-out cohorts
are deliberately un-pre-writable:

  - WITHHELD real codes: genuine NPCI / Razorpay codes that are NOT in the
    documented `CODE_MEANING` table and NOT in the trained classifier's training
    data. A lookup rule and a code-share model have no entry for them; only a
    model with world knowledge of payment errors can interpret them.
  - FREE-TEXT gateway messages: no structured code at all, phrased differently by
    different providers (retry hints, malformed-field descriptions, timeouts).
  - TWO-CAUSE blends: genuinely ambiguous → INSUFFICIENT_EVIDENCE.
  - RED HERRING: the dominant code is the ambiguous `U30`; the true signal is a
    sharply-elevated MINORITY code.

Every cohort's ground-truth cause is fixed in this file BEFORE any method runs
(TASK C: "Ground-truth labels for these must be written before any method is
run"). The diagnoser never imports this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from diagnose.schema import (
    CAUSE_CUSTOMER, CAUSE_ISSUER, CAUSE_MERCHANT, CAUSE_NETWORK, INSUFFICIENT,
)

# --- WITHHELD real codes (real; withheld from the table + training) -------------
# NPCI UPI: Z8 per-transaction limit (bank-set); U16 risk/fraud threshold.
# Razorpay reasons: bank_not_available, psp_app_not_available, gateway_technical_error.
WITHHELD_CODE_CAUSE = {
    "Z8": CAUSE_CUSTOMER,                    # per-txn limit set by the customer's bank
    "U16": CAUSE_CUSTOMER,                   # risk / suspected-fraud decline
    "bank_not_available": CAUSE_ISSUER,      # acquiring/issuing bank unavailable
    "psp_app_not_available": CAUSE_ISSUER,   # customer's bank PSP handle down
    "gateway_technical_error": CAUSE_NETWORK,  # transient gateway technical error
}

# --- FREE-TEXT gateway messages (no code) with their true cause -----------------
FREETEXT = {
    CAUSE_ISSUER: [
        "Issuer bank is currently unavailable, please retry after some time.",
        "Remitter bank is under maintenance; transactions temporarily declined.",
        "Beneficiary bank not responding to authorization requests.",
    ],
    CAUSE_NETWORK: [
        "Upstream acquirer connection timed out after 30s.",
        "Gateway did not receive a response from the network; please retry.",
        "Temporary network error while contacting the switch.",
    ],
    CAUSE_MERCHANT: [
        "Invalid value for field 'card_expiry' in the request.",
        "Mandatory parameter 'order_id' missing from the payment request.",
        "Amount is below the minimum permitted for this payment method.",
    ],
    CAUSE_CUSTOMER: [
        "Customer has insufficient balance in the account.",
        "Card reported lost or blocked by the cardholder.",
        "Customer did not complete authentication in time.",
    ],
    # deliberately vague — no attributable cause
    INSUFFICIENT: [
        "Transaction failed, please try again.",
        "Payment could not be completed at this time.",
        "An unexpected error occurred.",
    ],
}


@dataclass
class OpenCohort:
    label: str
    window: str
    true_cause: str
    kind: str                       # "clear" | "ambiguous"
    cohort_counts: dict = field(default_factory=dict)   # KNOWN codes only
    unknown_codes: dict = field(default_factory=dict)   # withheld real codes
    messages: tuple = ()            # free-text gateway strings


def _msgs(cause: str, n: int) -> tuple:
    pool = FREETEXT[cause]
    return tuple(pool[i % len(pool)] for i in range(n))


def build_open_world_cohorts() -> list[OpenCohort]:
    """The held-out open-world set. Sizes are >= the sample-size guardrail so every
    method actually engages; labels are fixed here, before any run."""
    C: list[OpenCohort] = []

    # 1-2. Cohorts dominated by a WITHHELD real code (lookup rule has no entry).
    C.append(OpenCohort("withheld: bank_not_available surge", "a 4h window",
                        CAUSE_ISSUER, "clear",
                        unknown_codes={"bank_not_available": 74}, cohort_counts={"Z9": 6}))
    C.append(OpenCohort("withheld: PSP-handle outage", "a 4h window",
                        CAUSE_ISSUER, "clear",
                        unknown_codes={"psp_app_not_available": 70}, cohort_counts={"U69": 8}))
    C.append(OpenCohort("withheld: per-txn bank limit (Z8)", "a 4h window",
                        CAUSE_CUSTOMER, "clear",
                        unknown_codes={"Z8": 61, "U16": 12}, cohort_counts={"Z9": 9}))
    C.append(OpenCohort("withheld: gateway_technical_error", "a 4h window",
                        CAUSE_NETWORK, "clear",
                        unknown_codes={"gateway_technical_error": 66}, cohort_counts={"Z9": 7}))

    # 3-5. FREE-TEXT cohorts (no structured code at all).
    C.append(OpenCohort("free-text: issuer-unavailable messages", "a 4h window",
                        CAUSE_ISSUER, "clear", messages=_msgs(CAUSE_ISSUER, 60)))
    C.append(OpenCohort("free-text: acquirer timeouts", "a 4h window",
                        CAUSE_NETWORK, "clear", messages=_msgs(CAUSE_NETWORK, 58)))
    C.append(OpenCohort("free-text: malformed-request messages", "a 4h window",
                        CAUSE_MERCHANT, "clear", messages=_msgs(CAUSE_MERCHANT, 55)))

    # 6. RED HERRING: dominant code is the ambiguous U30; the true signal is a
    #    sharply-elevated MINORITY known code (GATEWAY_ERROR, ~4x its baseline).
    C.append(OpenCohort("red herring: U30-dominant, network minority", "a 4h window",
                        CAUSE_NETWORK, "clear",
                        cohort_counts={"U30": 62, "GATEWAY_ERROR": 28, "Z9": 8}))

    # 7. TWO-CAUSE blend across a withheld issuer code and a known merchant code.
    C.append(OpenCohort("blend: withheld-issuer + merchant", "a 4h window",
                        INSUFFICIENT, "ambiguous",
                        cohort_counts={"BAD_REQUEST_ERROR": 40},
                        unknown_codes={"bank_not_available": 40}))

    # 8. FREE-TEXT with genuinely vague messages -> no attributable cause.
    C.append(OpenCohort("free-text: vague / uninformative", "a 4h window",
                        INSUFFICIENT, "ambiguous", messages=_msgs(INSUFFICIENT, 52)))

    return C
