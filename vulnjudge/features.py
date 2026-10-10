"""Feature engineering for VulnJudge.

Every feature is a plain number in [0, 1] with a human-readable meaning.
The model is a linear logistic regression, so each feature's contribution
to a prediction is simply weight * value — fully interpretable.
"""

# Canonical vulnerability classes. Unknown classes are mapped to the
# global base rate at score time (never crash on a new class name).
VULN_CLASSES = [
    "sqli",
    "auth_bypass",
    "idor",
    "ssrf",
    "rce",
    "xxe",
    "lfi",
    "xss",
    "info_disclosure",
    "business_logic",
    "csrf",
    "open_redirect",
]

FEATURE_NAMES = [
    "class_base_rate",   # learned P(valid | vuln_class) from training labels (Laplace-smoothed)
    "unauthenticated",   # 1.0 when the endpoint needs no auth — unauthenticated findings are stronger
    "has_poc",           # 1.0 when a proof-of-concept is attached
    "response_diff",     # 1.0 when the response observably differs between probe and control
    "sensitive_asset",   # high=1.0, medium=0.5, low=0.0
    "method_risk",       # 1.0 for state-changing methods (POST/PUT/PATCH/DELETE)
    "endpoint_depth",    # path segment count / 6, capped at 1.0 (deeper = more app logic)
    "status_2xx",        # 1.0 when the probe returned 2xx
    "is_api",            # 1.0 when asset type is "api"
    "evidence_strength", # (has_poc + response_diff) / 2 — combined evidence signal
]

FEATURE_DOCS = {
    "class_base_rate": "Historical validity rate of this vuln class in your labeled data.",
    "unauthenticated": "Endpoint reachable without authentication.",
    "has_poc": "A working proof-of-concept is attached to the finding.",
    "response_diff": "Probe vs control responses differ observably.",
    "sensitive_asset": "Asset sensitivity: high=1.0, medium=0.5, low=0.0.",
    "method_risk": "State-changing HTTP method (POST/PUT/PATCH/DELETE).",
    "endpoint_depth": "API path depth (segments/6, capped at 1.0).",
    "status_2xx": "Probe returned HTTP 2xx.",
    "is_api": "Asset is an API (vs web/mobile).",
    "evidence_strength": "Combined evidence: (has_poc + response_diff) / 2.",
}

_SENSITIVITY = {"high": 1.0, "medium": 0.5, "low": 0.0}
_STATE_CHANGING = {"POST", "PUT", "PATCH", "DELETE"}


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n", ""):
        return False
    return default


def fit_class_priors(rows, label_key="valid"):
    """Learn P(label | vuln_class) from labeled rows with Laplace smoothing.

    Returns (priors_dict, global_prior). Unseen classes fall back to the
    global prior at score time.
    """
    from collections import defaultdict

    pos = defaultdict(int)
    tot = defaultdict(int)
    for r in rows:
        cls = str(r.get("vuln_class", "")).strip().lower() or "__unknown__"
        tot[cls] += 1
        if _to_bool(r.get(label_key, False)):
            pos[cls] += 1
    priors = {c: (pos[c] + 1.0) / (tot[c] + 2.0) for c in tot}
    total_pos = sum(pos.values())
    total_n = sum(tot.values())
    global_prior = (total_pos + 1.0) / (total_n + 2.0) if total_n else 0.5
    return priors, global_prior


def _nested_get(finding, nested_key, flat_key, container="evidence", default=False):
    """Read a field that may live nested (score-time JSON) or flat (train-time CSV)."""
    container_d = finding.get(container) or {}
    if nested_key in container_d:
        return container_d[nested_key]
    return finding.get(flat_key, default)


def extract_features(finding, class_priors, global_prior):
    """Turn one finding dict into a feature vector (list of floats, [0,1]).

    Accepts both the nested score-time format (evidence: {...}, asset: {...})
    and the flat train-time CSV format (has_poc, asset_sensitivity, ... columns).
    """
    vuln_class = str(finding.get("vuln_class", "")).strip().lower()
    base_rate = class_priors.get(vuln_class, global_prior)

    auth_required = _to_bool(finding.get("auth_required", True), default=True)
    has_poc = _to_bool(_nested_get(finding, "has_poc", "has_poc"))
    response_diff = _to_bool(_nested_get(finding, "response_diff", "response_diff"))
    try:
        status_code = int(_nested_get(finding, "status_code", "status_code", default=0) or 0)
    except (TypeError, ValueError):
        status_code = 0

    asset = finding.get("asset") or {}
    sens_raw = asset.get("sensitivity", finding.get("asset_sensitivity", "low"))
    sensitivity = _SENSITIVITY.get(str(sens_raw).lower(), 0.0)
    type_raw = asset.get("type", finding.get("asset_type", ""))
    is_api = 1.0 if str(type_raw).lower() == "api" else 0.0

    method = str(finding.get("http_method", "GET")).upper()
    endpoint = str(finding.get("endpoint", "") or "")
    depth = len([p for p in endpoint.split("/") if p])

    return [
        float(base_rate),
        0.0 if auth_required else 1.0,
        1.0 if has_poc else 0.0,
        1.0 if response_diff else 0.0,
        float(sensitivity),
        1.0 if method in _STATE_CHANGING else 0.0,
        min(depth / 6.0, 1.0),
        1.0 if 200 <= status_code < 300 else 0.0,
        is_api,
        (float(has_poc) + float(response_diff)) / 2.0,
    ]


def feature_row(finding, class_priors, global_prior):
    """Feature vector paired with names: list of (name, value)."""
    return list(zip(FEATURE_NAMES, extract_features(finding, class_priors, global_prior)))
