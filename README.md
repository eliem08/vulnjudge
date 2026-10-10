# VulnJudge

A tiny **calibrated decision model for bug-bounty vulnerability triage** — "Jev for security."

Hunters drown in scanner noise. VulnJudge takes a finding and outputs a **calibrated P(valid vulnerability)** plus a severity estimate, with per-feature explanations and a triage recommendation (`chase` / `verify` / `deprioritize`). Pure Python, zero dependencies, logistic regression implemented from scratch.

## The problem

Triage is the bottleneck in bug bounty, not discovery. Scanners and recon pipelines surface hundreds of candidate findings; a human has to decide which ones deserve a proof-of-concept. That decision is repetitive, inconsistent, and burns the scarcest resource a hunter has: attention.

VulnJudge automates the *first pass*: given a finding's structured attributes (vuln class, auth, evidence, asset sensitivity…), it returns a calibrated probability the finding is real, so you chase the 0.9s and ignore the 0.05s.

## How it works

```
finding JSON → 10 interpretable features → logistic regression (from scratch)
             → bucket calibration on held-out data → P(valid), P(high_sev),
               top-3 contributing features, triage recommendation
```

- **Features → logreg → calibration.** Two models are trained: one for validity, one for high severity. Both are plain logistic regression (gradient descent + L2, implemented in `model.py` — no sklearn, no numpy).
- **Calibration is the credibility feature.** Raw model scores are mapped through buckets fit on a held-out split, and every `train` run prints a **reliability table**: predicted bucket vs. actual positive rate. If the table looks bad, you know not to trust the model. No black boxes.
- **Explainability is free.** Because the model is linear, each feature's contribution is exactly `weight × value`. Every score lists the top-3 drivers.

### Features

| Feature | Meaning |
|---|---|
| `class_base_rate` | Historical validity rate of this vuln class in *your* labeled data (Laplace-smoothed; unseen classes fall back to the global rate) |
| `unauthenticated` | Endpoint reachable without authentication |
| `has_poc` | A working proof-of-concept is attached |
| `response_diff` | Probe vs. control responses differ observably |
| `sensitive_asset` | Asset sensitivity: high=1.0, medium=0.5, low=0.0 |
| `method_risk` | State-changing HTTP method (POST/PUT/PATCH/DELETE) |
| `endpoint_depth` | API path depth (segments ÷ 6, capped at 1.0) |
| `status_2xx` | Probe returned HTTP 2xx |
| `is_api` | Asset is an API (vs web/mobile) |
| `evidence_strength` | Combined evidence: (`has_poc` + `response_diff`) ÷ 2 |

## Quickstart

```bash
pip install -e .
vulnjudge demo        # train on starter data, score examples, end-to-end
```

Or step by step:

```bash
# 1. Train on your labeled CSV (columns below), holding out 25% for calibration
vulnjudge train data/starter-labels.csv -o model.json
```

```
Training summary
  rows total ......... 60
  train / calibrate .. 45 / 15
  base rate (valid) .. 0.711
  base rate (high) ... 0.489
  Brier (valid, cal) . 0.0634
  Brier (high, cal) .. 0.0622
  acc@0.5 (valid) .... 0.933
  acc@0.5 (high) ..... 0.933

Reliability — P(valid) on held-out split
Bucket         n  Avg predicted  Actual rate
0.0-0.2        3          0.019        0.000 *
0.2-0.4        1          0.243        0.000
0.4-0.6        0          0.000        0.000
0.6-0.8        0          0.000        0.000
0.8-1.0       11          0.961        0.909 *
(* = bucket used for recalibration)
```

```bash
# 2. Score a finding
vulnjudge score examples/finding.json -m model.json
```

```
=== Blind SQLi in report export ===
class: sqli   endpoint: /api/reports/export   method: GET

P(valid vulnerability) : 0.909  [##################--]
P(high severity)       : 1.000  [####################]
confidence             : high
triage                 : CHASE

Top contributing features (validity model):
  + has_poc            value=1.00  contribution=+5.064
  + evidence_strength  value=1.00  contribution=+2.982
  - is_api             value=1.00  contribution=-1.965

Why: Strongest signals: has_poc, evidence_strength push toward valid; recommendation is 'chase'.
```

The numbers above are the **actual** held-out results on the 60-row starter dataset — see the honesty section below before quoting them anywhere.

### Finding JSON format

```json
{
  "title": "Blind SQLi in report export",
  "vuln_class": "sqli",
  "endpoint": "/api/reports/export",
  "auth_required": true,
  "http_method": "GET",
  "evidence": {"has_poc": true, "response_diff": true, "status_code": 200},
  "asset": {"type": "api", "sensitivity": "high"}
}
```

`score` accepts a single object or a list. Supported `vuln_class` values: `sqli`, `auth_bypass`, `idor`, `ssrf`, `rce`, `xxe`, `lfi`, `xss`, `info_disclosure`, `business_logic`, `csrf`, `open_redirect` (unknown classes fall back to the global base rate — never crash).

### Training CSV format

Columns: `title, vuln_class, endpoint, auth_required, http_method, has_poc, response_diff, status_code, asset_type, asset_sensitivity, valid, high_severity` — where `valid` and `high_severity` are the labels (`true`/`false`).

## Honesty about data

**The bundled `data/starter-labels.csv` (60 rows) is illustrative starter data, not real.** It was hand-written to demonstrate the pipeline: the classes are clean, the evidence signals are strong, and 15 held-out points make the reliability table look better than real life. Do not quote the 0.93 accuracy as a product claim — it measures "the demo works," nothing more.

**The real value comes from training on your own labeled data.** Suggested path:

1. **HackerOne Hacktivity / disclosed reports** — hundreds of public disclosures with vuln class, severity, and often reproduction detail. Label: `valid=true` for accepted reports; mine your own *rejected* submissions (or N/A duplicates) for `valid=false`. That's the highest-signal negative class you own.
2. **Your scanner output** — every finding your recon pipeline ever produced, labeled by what you actually pursued vs. dropped.
3. **Program-specific calibration** — retrain per program once you have 100+ labels; different programs have wildly different bar heights, and the reliability table will show it.

Aim for 200+ labeled rows before trusting the calibration buckets; below that, treat P(valid) as a *ranking* signal (sort your queue by it) rather than a true probability.

## Calibration, explained

A model that says "0.9" should be right 9 times out of 10. `train` holds out 25% of your labels, fits 5 probability buckets on them, and prints predicted-vs-actual rates per bucket. New predictions are then mapped to their bucket's empirical rate (buckets with fewer than 2 points fall back to the raw score — small data stays honest).

Watch the table, not just accuracy: a model can be 90% accurate and still be miscalibrated (e.g., everything predicted at 0.6). If a bucket's actual rate diverges from its predicted average, get more labels for that region.

## Limitations

- Linear model: it cannot learn feature interactions (e.g., "unauthenticated *and* sensitive asset" is just the sum of both). v0.1 trades expressiveness for interpretability.
- Garbage in, garbage out: trained on biased labels, it reproduces your biases — including which vuln classes you historically bothered to report.
- Not a vulnerability detector: it triages *candidate findings you already have*. It will not find bugs for you.
- Small-data mode: under ~200 labels, use the scores to *rank*, not to auto-close.

## Roadmap

- Per-program calibration profiles (`vulnjudge train --program uber`)
- Active learning: `vulnjudge suggest` picks the findings whose labels would most improve the model
- Interaction features (optional, opt-in — keeps v0.1's interpretability as default)
- ONNX/export for embedding in recon pipelines

## License

MIT — © 2026 Ilyass Emssaad
