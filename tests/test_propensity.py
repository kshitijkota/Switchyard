"""§9 / ABSOLUTE RULE 2: propensities are recorded at decision time (equal to the
policy's own probability of the chosen action) and never reconstructed."""

from __future__ import annotations

import numpy as np

from events import PROCESSORS
from policy.explore import EPSILON, EpsilonGreedyPolicy
from policy.legacy import LegacyPolicy, legacy_mode, probs
from sim.traffic import generate_traffic


def test_legacy_propensity_equals_policy_probability():
    rng = np.random.default_rng(0)
    legacy = LegacyPolicy()
    for ctx in list(generate_traffic(2000, seed=5)):
        d = legacy.decide(ctx, rng)
        assert d.propensity == probs(ctx)[d.processor]      # recorded at decision time
        assert 0.0 < d.propensity <= 1.0


def test_exploration_min_propensity_is_epsilon_over_three():
    policy = EpsilonGreedyPolicy(exploit_proc_fn=legacy_mode, epsilon=EPSILON)
    rng = np.random.default_rng(1)
    seen_min = 1.0
    for ctx in list(generate_traffic(3000, seed=6)):
        d, was_explore, exploit = policy.decide(ctx, rng)
        # propensity matches the mixture formula at decision time
        expected = (1 - EPSILON) * (1.0 if d.processor == exploit else 0.0) + EPSILON / len(PROCESSORS)
        assert abs(d.propensity - expected) < 1e-12
        seen_min = min(seen_min, d.propensity)
    assert abs(seen_min - EPSILON / len(PROCESSORS)) < 1e-12   # 0.01


def test_estimators_do_not_reconstruct_propensity():
    """Structural guard: estimator code never calls a policy to recompute a
    propensity — it only reads the logged value. Scan every module under
    estimators/ for references to the policy probability functions or an import
    of the legacy policy."""
    import ast, os
    est_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "estimators")
    for root, _, files in os.walk(est_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            tree = ast.parse(open(os.path.join(root, f)).read())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            assert "probs" not in names and "legacy_mode" not in names, f"{f} recomputes propensity"
            # must not import the propensity-defining policies (legacy/explore)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "") in ("policy.legacy", "policy.explore"):
                    raise AssertionError(f"{f} imports a logging policy to reconstruct propensity")
