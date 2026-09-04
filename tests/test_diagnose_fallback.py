"""§8: malformed provider output must fall back to INSUFFICIENT_EVIDENCE (never a
wrong assertion) and log an incident; abstention on tiny cohorts."""

from __future__ import annotations

from diagnose.node import Diagnoser, DiagnosisInput, StatisticalProvider
from diagnose.schema import INSUFFICIENT


class _BrokenProvider:
    name = "broken"

    def diagnose(self, inp):
        return {"cause": "NOT_A_REAL_CAUSE", "confidence": 5}  # fails schema


class _GarbageProvider:
    name = "garbage"

    def diagnose(self, inp):
        return "totally not json"


def _inp():
    return DiagnosisInput("c", "w", {"U30": 50, "U69": 10}, {"U30": 12, "U69": 55})


def test_schema_failure_falls_back(tmp_path):
    d = Diagnoser(provider=_BrokenProvider(), cache_dir=str(tmp_path))
    out = d.diagnose(_inp(), use_cache=False)
    assert out["cause"] == INSUFFICIENT and out["confidence"] == 0.0


def test_non_dict_output_falls_back(tmp_path):
    d = Diagnoser(provider=_GarbageProvider(), cache_dir=str(tmp_path))
    out = d.diagnose(_inp(), use_cache=False)
    assert out["cause"] == INSUFFICIENT


def test_statistical_abstains_on_tiny_cohort(tmp_path):
    d = Diagnoser(provider=StatisticalProvider(), cache_dir=str(tmp_path))
    out = d.diagnose(DiagnosisInput("tiny", "w", {"U30": 5, "U69": 3},
                                    {"U30": 12, "U69": 55}), use_cache=False)
    assert out["cause"] == INSUFFICIENT   # too few failures to conclude


def test_statistical_detects_clear_issuer_spike(tmp_path):
    d = Diagnoser(provider=StatisticalProvider(), cache_dir=str(tmp_path))
    out = d.diagnose(DiagnosisInput("issuer", "w", {"U30": 70, "U69": 10, "GATEWAY_ERROR": 5},
                                    {"U30": 12, "U69": 55, "GATEWAY_ERROR": 20}), use_cache=False)
    assert out["cause"] == "ISSUER_DEGRADATION"
