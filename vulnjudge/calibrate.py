"""Bucket calibration on a held-out split + reliability reporting.

A raw logistic-regression score is not a probability you can trust until it
is calibrated: in each predicted-probability bucket, the *actual* positive
rate should match the *average predicted* probability. This module builds
the buckets on held-out data and can map new predictions to their bucket's
empirical rate (bucket recalibration).
"""

N_BUCKETS = 5


def bucket_edges(n_buckets=N_BUCKETS):
    step = 1.0 / n_buckets
    return [(round(i * step, 4), round((i + 1) * step, 4)) for i in range(n_buckets)]


def bucket_index(p, n_buckets=N_BUCKETS):
    i = int(p * n_buckets)
    return min(max(i, 0), n_buckets - 1)


def fit_buckets(y_true, y_prob, n_buckets=N_BUCKETS, min_n=2):
    """Fit bucket recalibration on held-out (y_true, y_prob) pairs.

    Returns a list of dicts: {lo, hi, n, mean_pred, actual_rate, calibrated}.
    `calibrated` is the empirical rate when the bucket has >= min_n points,
    else None (fall back to the raw probability at score time).
    """
    edges = bucket_edges(n_buckets)
    buckets = [{"lo": lo, "hi": hi, "n": 0, "pred_sum": 0.0, "pos": 0}
               for lo, hi in edges]
    for y, p in zip(y_true, y_prob):
        b = buckets[bucket_index(p, n_buckets)]
        b["n"] += 1
        b["pred_sum"] += p
        b["pos"] += int(bool(y))
    out = []
    for b in buckets:
        n = b["n"]
        mean_pred = b["pred_sum"] / n if n else 0.0
        actual = b["pos"] / n if n else 0.0
        out.append({
            "lo": b["lo"], "hi": b["hi"], "n": n,
            "mean_pred": round(mean_pred, 4),
            "actual_rate": round(actual, 4),
            "calibrated": round(actual, 4) if n >= min_n else None,
        })
    return out


def apply_buckets(p, buckets, n_buckets=N_BUCKETS):
    """Map a raw probability to its bucket's empirical rate (if reliable)."""
    b = buckets[bucket_index(p, n_buckets)]
    return b["calibrated"] if b["calibrated"] is not None else p


def brier_score(y_true, y_prob):
    n = len(y_true)
    if not n:
        return 0.0
    return sum((p - float(y)) ** 2 for y, p in zip(y_true, y_prob)) / n


def accuracy_at_threshold(y_true, y_prob, threshold=0.5):
    n = len(y_true)
    if not n:
        return 0.0
    return sum(1 for y, p in zip(y_true, y_prob)
               if (p >= threshold) == bool(y)) / n
