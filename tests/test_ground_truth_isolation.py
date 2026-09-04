"""AGENT_BRIEF §4.2 / §9: sim.ground_truth must never be importable from
policy/ or estimators/ (and, for good measure, diagnose/node.py — the LLM node
must not see latent truth either). Walk the AST of every module and fail if the
import appears. Do not weaken this test."""

from __future__ import annotations

import ast
import os

_ROOT = os.path.dirname(os.path.dirname(__file__))
GUARDED_DIRS = ["policy", "estimators"]
GUARDED_FILES = [os.path.join("diagnose", "node.py"), os.path.join("diagnose", "schema.py")]
FORBIDDEN = "sim.ground_truth"


def _python_files():
    for d in GUARDED_DIRS:
        for root, _, files in os.walk(os.path.join(_ROOT, d)):
            for f in files:
                if f.endswith(".py"):
                    yield os.path.join(root, f)
    for f in GUARDED_FILES:
        yield os.path.join(_ROOT, f)


def _imports_ground_truth(path: str) -> bool:
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == FORBIDDEN or alias.name.startswith(FORBIDDEN + "."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == FORBIDDEN or mod.startswith(FORBIDDEN + "."):
                return True
            # catch `from sim import ground_truth`
            if mod == "sim" and any(a.name == "ground_truth" for a in node.names):
                return True
    return False


def test_no_ground_truth_import_in_guarded_code():
    offenders = [p for p in _python_files() if _imports_ground_truth(p)]
    assert not offenders, f"ground truth leaked into: {offenders}"


def test_guard_actually_scans_files():
    # Sanity: the walk finds real modules (so the test can't pass vacuously).
    assert sum(1 for _ in _python_files()) >= 6
