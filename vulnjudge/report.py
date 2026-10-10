"""Console report formatting. All presentation lives here; logic stays
in model.py / calibrate.py / features.py."""


def _bar(p, width=20):
    filled = int(round(p * width))
    return "#" * filled + "-" * (width - filled)


def format_reliability_table(title, buckets):
    lines = [title, f"{'Bucket':<12}{'n':>4}  {'Avg predicted':>13}  {'Actual rate':>11}"]
    for b in buckets:
        label = f"{b['lo']:.1f}-{b['hi']:.1f}"
        cal = "" if b["calibrated"] is None else " *"
        lines.append(f"{label:<12}{b['n']:>4}  {b['mean_pred']:>13.3f}  {b['actual_rate']:>11.3f}{cal}")
    lines.append("(* = bucket used for recalibration)")
    return "\n".join(lines)


def format_score(result):
    f = result["finding"]
    lines = [
        f"=== {f.get('title', '(untitled)')} ===",
        f"class: {f.get('vuln_class', '?')}   endpoint: {f.get('endpoint', '?')}   "
        f"method: {f.get('http_method', '?')}",
        "",
        f"P(valid vulnerability) : {result['p_valid']:.3f}  [{_bar(result['p_valid'])}]",
        f"P(high severity)       : {result['p_high']:.3f}  [{_bar(result['p_high'])}]",
        f"confidence             : {result['confidence']}",
        f"triage                 : {result['recommendation'].upper()}",
        "",
        "Top contributing features (validity model):",
    ]
    for name, value, contrib in result["top_features"]:
        arrow = "+" if contrib >= 0 else "-"
        lines.append(f"  {arrow} {name:<18} value={value:.2f}  contribution={contrib:+.3f}")
    lines.append("")
    lines.append(f"Why: {result['explanation']}")
    return "\n".join(lines)


def format_training_summary(stats):
    lines = [
        "Training summary",
        f"  rows total ......... {stats['n_total']}",
        f"  train / calibrate .. {stats['n_train']} / {stats['n_cal']}",
        f"  base rate (valid) .. {stats['base_rate_valid']:.3f}",
        f"  base rate (high) ... {stats['base_rate_high']:.3f}",
        f"  Brier (valid, cal) . {stats['brier_valid']:.4f}",
        f"  Brier (high, cal) .. {stats['brier_high']:.4f}",
        f"  acc@0.5 (valid) .... {stats['acc_valid']:.3f}",
        f"  acc@0.5 (high) ..... {stats['acc_high']:.3f}",
    ]
    return "\n".join(lines)
