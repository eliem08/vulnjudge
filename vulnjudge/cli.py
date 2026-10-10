"""VulnJudge command-line interface."""

import argparse
import csv
import json
import os
import random
import sys

from .features import (
    FEATURE_NAMES,
    _to_bool,
    extract_features,
    fit_class_priors,
)
from .model import LogisticRegression, load_model, save_model
from .calibrate import (
    N_BUCKETS,
    accuracy_at_threshold,
    apply_buckets,
    brier_score,
    fit_buckets,
)
from .report import (
    format_reliability_table,
    format_score,
    format_training_summary,
)


def _package_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def load_labeled_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_rows(rows, cal_fraction=0.25, seed=42):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    n_cal = max(1, int(round(len(rows) * cal_fraction)))
    cal_idx = set(idx[:n_cal])
    train = [r for i, r in enumerate(rows) if i not in cal_idx]
    cal = [r for i, r in enumerate(rows) if i in cal_idx]
    return train, cal


def train_models(train_rows, cal_rows):
    """Fit class priors + validity model + severity model; calibrate both."""
    priors, global_prior = fit_class_priors(train_rows, "valid")

    def feats(rows):
        return [extract_features(r, priors, global_prior) for r in rows]

    X_train, X_cal = feats(train_rows), feats(cal_rows)
    y_valid_train = [1 if _to_bool(r.get("valid", False)) else 0 for r in train_rows]
    y_high_train = [1 if _to_bool(r.get("high_severity", False)) else 0 for r in train_rows]
    y_valid_cal = [1 if _to_bool(r.get("valid", False)) else 0 for r in cal_rows]
    y_high_cal = [1 if _to_bool(r.get("high_severity", False)) else 0 for r in cal_rows]

    m_valid = LogisticRegression().fit(X_train, y_valid_train)
    m_high = LogisticRegression().fit(X_train, y_high_train)

    p_valid_cal = m_valid.predict_proba(X_cal)
    p_high_cal = m_high.predict_proba(X_cal)
    buckets_valid = fit_buckets(y_valid_cal, p_valid_cal)
    buckets_high = fit_buckets(y_high_cal, p_high_cal)

    payload = {
        "version": "0.1.0",
        "feature_names": FEATURE_NAMES,
        "class_priors": priors,
        "global_prior": global_prior,
        "n_buckets": N_BUCKETS,
        "validity": {
            "model": m_valid.to_dict(),
            "buckets": buckets_valid,
        },
        "severity": {
            "model": m_high.to_dict(),
            "buckets": buckets_high,
        },
    }
    stats = {
        "n_total": len(train_rows) + len(cal_rows),
        "n_train": len(train_rows),
        "n_cal": len(cal_rows),
        "base_rate_valid": sum(y_valid_train) / len(y_valid_train),
        "base_rate_high": sum(y_high_train) / len(y_high_train),
        "brier_valid": brier_score(y_valid_cal, p_valid_cal),
        "brier_high": brier_score(y_high_cal, p_high_cal),
        "acc_valid": accuracy_at_threshold(y_valid_cal, p_valid_cal),
        "acc_high": accuracy_at_threshold(y_high_cal, p_high_cal),
    }
    return payload, stats


def confidence_bucket(p):
    if p <= 0.25 or p >= 0.75:
        return "high"
    if p <= 0.35 or p >= 0.65:
        return "medium"
    return "low"


def recommend(p_valid, p_high):
    if p_valid >= 0.70:
        return "chase"
    if p_valid >= 0.55 and p_high >= 0.55:
        return "chase"
    if p_valid >= 0.40:
        return "verify"
    return "deprioritize"


def score_finding(finding, payload):
    priors = payload["class_priors"]
    global_prior = payload["global_prior"]
    n_buckets = payload.get("n_buckets", N_BUCKETS)
    x = extract_features(finding, priors, global_prior)

    mv = LogisticRegression.from_dict(payload["validity"]["model"])
    mh = LogisticRegression.from_dict(payload["severity"]["model"])
    p_valid_raw = mv.predict_proba(x)
    p_high_raw = mh.predict_proba(x)
    p_valid = apply_buckets(p_valid_raw, payload["validity"]["buckets"], n_buckets)
    p_high = apply_buckets(p_high_raw, payload["severity"]["buckets"], n_buckets)

    contribs = mv.contributions(x)
    ranked = sorted(
        ((FEATURE_NAMES[i], x[i], c) for i, c in contribs),
        key=lambda t: abs(t[2]),
        reverse=True,
    )
    top3 = [(name, val, round(c, 4)) for name, val, c in ranked[:3]]

    rec = recommend(p_valid, p_high)
    pos = [n for n, _, c in top3 if c > 0]
    explanation = (
        f"Strongest signals: {', '.join(pos) if pos else 'none'} push toward valid; "
        f"recommendation is '{rec}'."
    )
    return {
        "finding": finding,
        "p_valid": round(p_valid, 4),
        "p_high": round(p_high, 4),
        "p_valid_raw": round(p_valid_raw, 4),
        "confidence": confidence_bucket(p_valid),
        "recommendation": rec,
        "top_features": top3,
        "explanation": explanation,
    }


def load_findings(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [data]
    return list(data)


def cmd_train(args):
    rows = load_labeled_csv(args.labels)
    if not rows:
        print("error: no rows in labels file", file=sys.stderr)
        return 1
    train_rows, cal_rows = split_rows(rows, args.cal_split, args.seed)
    payload, stats = train_models(train_rows, cal_rows)
    save_model(args.output, payload)
    print(format_training_summary(stats))
    print()
    print(format_reliability_table("Reliability — P(valid) on held-out split",
                                   payload["validity"]["buckets"]))
    print()
    print(format_reliability_table("Reliability — P(high severity) on held-out split",
                                   payload["severity"]["buckets"]))
    print()
    print(f"model saved to {args.output}")
    return 0


def cmd_score(args):
    payload = load_model(args.model)
    findings = load_findings(args.finding)
    for i, finding in enumerate(findings):
        if i:
            print()
        print(format_score(score_finding(finding, payload)))
    return 0


def cmd_demo(args):
    import tempfile
    base = _package_dir()
    labels = os.path.join(base, "data", "starter-labels.csv")
    findings_path = os.path.join(base, "examples", "finding.json")
    if not os.path.exists(labels):
        print(f"error: starter data not found at {labels}", file=sys.stderr)
        return 1
    fd, model_path = tempfile.mkstemp(prefix="vulnjudge-demo-", suffix=".json")
    os.close(fd)
    try:
        print("### 1. Training on bundled starter data (ILLUSTRATIVE ONLY)\n")
        rc = cmd_train(argparse.Namespace(labels=labels, output=model_path,
                                          cal_split=0.25, seed=42))
        if rc:
            return rc
        print("\n### 2. Scoring example findings\n")
        return cmd_score(argparse.Namespace(finding=findings_path, model=model_path))
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="vulnjudge",
        description="Calibrated tiny decision model for bug-bounty triage.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train on a labeled CSV, save model JSON")
    t.add_argument("labels", help="CSV with finding columns + valid,high_severity labels")
    t.add_argument("-o", "--output", default="model.json")
    t.add_argument("--cal-split", type=float, default=0.25,
                   help="fraction held out for calibration (default 0.25)")
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("score", help="score finding(s) from a JSON file")
    s.add_argument("finding", help="JSON object or list of finding objects")
    s.add_argument("-m", "--model", default="model.json")
    s.set_defaults(func=cmd_score)

    d = sub.add_parser("demo", help="train on starter data and score examples end-to-end")
    d.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
