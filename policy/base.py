"""Shared interfaces for logging policies and learned routing methods.

NOTE: nothing in policy/ or estimators/ may import sim.ground_truth (enforced by
tests/test_ground_truth_isolation.py). These interfaces see only contexts and
logged events — never latent truth.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from events import Context, Decision


@runtime_checkable
class LoggingPolicy(Protocol):
    """A policy that turns a context into a Decision, recording the propensity of
    the chosen processor AT DECISION TIME (never reconstructed afterwards)."""

    policy_version: str

    def decide(self, context: Context, rng: np.random.Generator) -> Decision:
        ...


@runtime_checkable
class RoutingMethod(Protocol):
    """The common interface for the four learned methods (direct/ips/dr/switchyard).

    - fit() learns from logged events.
    - recommend() returns the chosen processor for a context (the policy).
    - estimated_value_per_1k() is the method's OWN estimate of its policy value.
    Ground-truth ("true") value is computed by the eval harness, not here.
    """

    name: str

    def recommend(self, context: Context) -> str:
        ...

    def estimated_value_per_1k(self) -> float:
        ...
