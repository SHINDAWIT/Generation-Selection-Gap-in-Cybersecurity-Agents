# K=25 paper-ready result summary

## Status and design

This exploratory post-hoc extension applies the complete selector and
abstention pipeline to 25 ordered Granite 4.2 8B Q4_K_M rollouts for each of 15
static-artifact CyBench tasks (375 real model records). It reuses the exact
K=25 prefix of the previously acquired K=50 evidence. The parent task is the
statistical unit; intervals use 2,000 seeded
task-level bootstrap resamples that keep all 25 rollouts together.

## Main results

| Selector | Coverage | Selective accuracy | Overall accuracy | Oracle gap |
|---|---:|---:|---:|---:|
| Single | 1.000 | 0.133 | 0.133 | 0.333 |
| Normalized majority | 1.000 | 0.267 | 0.267 | 0.200 |
| Tool-grounded majority | 0.867 | 0.231 | 0.200 | 0.267 |
| Agreement abstention (≥15/25) | 0.467 | 0.429 | 0.200 | 0.267 |
| Oracle Pass@25 | 0.467 | 1.000 | 0.467 | 0.000 |

Normalized majority changed overall accuracy by
+0.133 versus Single.
Its paired bootstrap delta mean was +0.134 with a 95%
percentile interval [+0.000,
+0.333]. Oracle Pass@25 exceeded Single by a bootstrap
mean of +0.333 [+0.133,
+0.600].

## Abstention and risk

The fixed ≥15/25 agreement rule covered 0.467 of tasks at
0.429 selective accuracy. Descriptive support
thresholds 10/25, 15/25, 20/25, and 25/25 yielded coverage/risk pairs of
(0.867, 0.692),
(0.467, 0.571),
(0.333, 0.600), and
(0.200, 0.333), respectively.
The wrong-consensus rate among ≥15/25 majority decisions was
0.571.

## Failure accounting and interpretation

Final-answer completion, parse-failure, tool-failure, and infrastructure-
failure rates were 0.931,
0.069, 0.125, and
0.000. The 375 records generated
92045 tokens in
1427.9 aggregate wall-clock seconds.

Oracle Pass@25 measures reachability with gold and cannot be deployed as a
selector. Exact normalized majority may split semantically equivalent verbose
answers, while high agreement can consistently favor an incorrect answer.
Because the K=50 curve was already inspected before this analysis, these
results are suitable as an exploratory extension and failure analysis, not an
independent confirmatory test or an absolute cybersecurity-capability claim.

## Reproducible assets

Raw evidence is `data/raw/main_k25_rollouts.jsonl`; decisions are
`data/processed/k25_decisions.json`; complete metrics and bootstrap intervals
are `results/k25_metrics.json`; the risk-coverage figure is
`results/figures/k25_risk_coverage.svg`; hashes are in
`results/k25_provenance.json`.
