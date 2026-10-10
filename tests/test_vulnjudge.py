"""Tests for VulnJudge. Run with: python3 -m pytest tests/ -q"""

import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vulnjudge.features import (
    FEATURE_NAMES,
    _to_bool,
    extract_features,
    fit_class_priors,
)
from vulnjudge.model import LogisticRegression, sigmoid
from vulnjudge.calibrate import (
    apply_buckets,
    brier_score,
    bucket_index,
    fit_buckets,
)
from vulnjudge.cli import recommend, score_finding, train_models

BASE = os.path.join(os.path.dirname(__file__), "..")
LABELS = os.path.join(BASE, "data", "starter-labels.csv")
EXAMPLES = os.path.join(BASE, "examples", "finding.json")


def _rows():
    import csv

    with open(LABELS, newline="") as f:
        return list(csv.DictReader(f))


def test_sigmoid_extremes():
    assert abs(sigmoid(1000) - 1.0) < 1e-9
    assert abs(sigmoid(-1000) - 0.0) < 1e-9
    assert abs(sigmoid(0) - 0.5) < 1e-9


def test_logreg_converges_on_separable_data():
    X = [[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]]
    y = [0, 0, 0, 1, 1, 1]
    m = LogisticRegression(lr=1.0, iters=2000, seed=1).fit(X, y)
    probs = m.predict_proba(X)
    assert all(p < 0.3 for p in probs[:3])
    assert all(p > 0.7 for p in probs[3:])


def test_logreg_save_load_roundtrip(tmp_path):
    X = [[0.0, 1.0], [1.0, 0.0]]
    y = [0, 1]
    m = LogisticRegression(iters=500, seed=7).fit(X, y)
    d = m.to_dict()
    m2 = LogisticRegression.from_dict(d)
    assert m2.predict_proba(X) == pytest.approx(m.predict_proba(X))


def test_to_bool_parsing():
    assert _to_bool("true") is True
    assert _to_bool("FALSE") is False
    assert _to_bool("1") is True
    assert _to_bool("0") is False
    assert _to_bool(None, default=True) is True
    assert _to_bool("yes") is True


def test_feature_extraction_values():
    priors = {"sqli": 0.8}
    f = {
        "vuln_class": "sqli",
        "endpoint": "/api/users/123/orders",
        "auth_required": False,
        "http_method": "POST",
        "evidence": {"has_poc": True, "response_diff": True, "status_code": 200},
        "asset": {"type": "api", "sensitivity": "high"},
    }
    x = extract_features(f, priors, 0.5)
    assert len(x) == len(FEATURE_NAMES)
    assert all(0.0 <= v <= 1.0 for v in x)
    d = dict(zip(FEATURE_NAMES, x))
    assert d["class_base_rate"] == pytest.approx(0.8)
    assert d["unauthenticated"] == 1.0
    assert d["has_poc"] == 1.0
    assert d["response_diff"] == 1.0
    assert d["sensitive_asset"] == 1.0
    assert d["method_risk"] == 1.0
    assert d["endpoint_depth"] == pytest.approx(4 / 6)
    assert d["status_2xx"] == 1.0
    assert d["is_api"] == 1.0
    assert d["evidence_strength"] == 1.0


def test_unknown_class_falls_back_to_global_prior():
    x = extract_features({"vuln_class": "brand_new_class"}, {}, 0.42)
    assert dict(zip(FEATURE_NAMES, x))["class_base_rate"] == pytest.approx(0.42)


def test_class_priors_smoothing():
    rows = [
        {"vuln_class": "sqli", "valid": "true"},
        {"vuln_class": "sqli", "valid": "true"},
        {"vuln_class": "xss", "valid": "false"},
    ]
    priors, glob = fit_class_priors(rows)
    assert priors["sqli"] == pytest.approx(3 / 4)  # (2+1)/(2+2)
    assert priors["xss"] == pytest.approx(1 / 3)  # (0+1)/(1+2)
    assert glob == pytest.approx(3 / 5)  # (2+1)/(3+2)


def test_starter_data_loads_with_60_rows():
    rows = _rows()
    assert len(rows) == 60
    assert all("valid" in r and "high_severity" in r for r in rows)


def test_train_models_produces_payload_and_stats():
    rows = _rows()
    payload, stats = train_models(rows[:45], rows[45:])
    assert payload["feature_names"] == FEATURE_NAMES
    assert len(payload["validity"]["model"]["weights"]) == len(FEATURE_NAMES)
    assert len(payload["validity"]["buckets"]) == 5
    assert 0.0 <= stats["brier_valid"] <= 1.0
    assert stats["n_total"] == 60


def test_calibration_buckets_structure():
    y = [1, 1, 0, 0, 1, 0]
    p = [0.9, 0.8, 0.1, 0.3, 0.6, 0.4]
    buckets = fit_buckets(y, p, n_buckets=2)
    assert len(buckets) == 2
    assert buckets[0]["n"] + buckets[1]["n"] == 6
    assert bucket_index(0.99, 2) == 1
    assert bucket_index(0.0, 2) == 0
    # bucket 1 (high probs): 0.9,0.8,0.6 -> all positive
    assert buckets[1]["actual_rate"] == pytest.approx(1.0)


def test_apply_buckets_uses_empirical_rate():
    buckets = fit_buckets([1, 1, 0, 0], [0.9, 0.85, 0.1, 0.15], n_buckets=2)
    assert apply_buckets(0.95, buckets, 2) == pytest.approx(1.0)
    assert apply_buckets(0.05, buckets, 2) == pytest.approx(0.0)


def test_brier_score_known():
    assert brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_recommend_thresholds():
    assert recommend(0.9, 0.9) == "chase"
    assert recommend(0.6, 0.7) == "chase"
    assert recommend(0.5, 0.2) == "verify"
    assert recommend(0.1, 0.1) == "deprioritize"


def test_score_finding_end_to_end():
    rows = _rows()
    payload, _ = train_models(rows[:45], rows[45:])
    with open(EXAMPLES) as f:
        findings = json.load(f)
    assert len(findings) == 3
    results = [score_finding(f, payload) for f in findings]
    for r in results:
        assert 0.0 <= r["p_valid"] <= 1.0
        assert 0.0 <= r["p_high"] <= 1.0
        assert r["recommendation"] in ("chase", "verify", "deprioritize")
        assert len(r["top_features"]) == 3
    # the strong SQLi finding should outscore the weak open redirect
    assert results[0]["p_valid"] > results[1]["p_valid"]


def test_cli_demo_runs(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = BASE + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        [sys.executable, "-m", "vulnjudge.cli", "demo"],
        capture_output=True, text=True, cwd=BASE, env=env, timeout=120,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Reliability" in r.stdout
    assert "triage" in r.stdout
