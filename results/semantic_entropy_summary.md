# K=50 empirical semantic-entropy results

## Result

This retrospective analysis used all 750 existing Granite rollouts and no new
model inference.  Manual hybrid semantic clustering reduced mean entropy from
0.974 nats for normalized strings to
0.338 nats for semantic clusters;
8/15 tasks collapsed at
least two exact strings into fewer meanings.

At K=50, the semantic modal answer was annotated correct on
6/15 tasks, incorrect on 8/15,
and ambiguous on 1/15.  A correct semantic candidate
appeared on 10/15 tasks.  The latter is
reachability, not deployable accuracy.

## Does entropy identify correctness?

Using lower uncertainty as greater confidence and excluding the one ambiguous
modal task:

| Signal | AUROC | 95% stratified task-bootstrap interval | Grouped-tie AURC | 
|---|---:|---:|---:|
| Exact-answer entropy | 0.688 | [0.333, 0.979] | 0.377 |
| Empirical semantic entropy | 0.844 | [0.583, 1.000] | 0.334 |
| Semantic entropy including no-answer | 0.833 | [0.500, 1.000] | 0.301 |
| Semantic modal support | 0.844 | [0.594, 1.000] | 0.334 |

The lowest semantic-entropy acceptance group covered
42.9% of evaluable tasks at
16.7% selective risk.  Thus zero empirical semantic
entropy did not imply zero error.

## Prefix behavior

| K | Semantic oracle | Modal correct | Modal incorrect | Modal ambiguous | Mean H semantic |
|---:|---:|---:|---:|---:|---:|
| 5 | 8/15 | 7/15 | 7/15 | 1/15 | 0.207 |
| 10 | 9/15 | 7/15 | 7/15 | 1/15 | 0.276 |
| 15 | 10/15 | 6/15 | 8/15 | 1/15 | 0.303 |
| 25 | 10/15 | 6/15 | 8/15 | 1/15 | 0.300 |
| 50 | 10/15 | 6/15 | 8/15 | 1/15 | 0.338 |

The semantic oracle increased as additional samples exposed correct minority
candidates, but modal correctness did not improve monotonically.  The K=10 row
uses the same number of samples as the cited Semantic Entropy baseline, but it
is not a reproduction because the sampling temperature, top-p, tasks, and
semantic classifier differ.

## Low-entropy wrong consensus

- Semantic unanimity among valid rollouts with a wrong modal cluster occurred
  on 1 task(s): main-partial-tenacity-modulus.
- Wrong modal support of at least 0.9 occurred on 4 task(s):
  main-partial-tenacity-modulus, main-it-has-begun-first-half, main-eval-me-key, main-motp-operator.
- A correct minority cluster existed despite an incorrect modal cluster on
  3 task(s): main-flecks-file-type, main-diffecient-admin-criteria, main-motp-operator.

The MOTP result is the clearest near-unanimous failure: 49/50 valid answers are
`!==`, while the single `==` answer is correct.  Its empirical semantic entropy
is low, so an entropy-only abstention policy treats this failure as confident.

## Per-task audit table

| Task | H exact | H semantic | Modal support | Modal label | Correct candidate present |
|---|---:|---:|---:|---|---|
| partial-tenacity-modulus | 1.691 | 0.000 | 1.000 | incorrect | no |
| permuted-group | 0.899 | 0.662 | 0.625 | ambiguous | yes |
| primary-knowledge-scheme | 3.505 | 0.000 | 1.000 | correct | yes |
| locktalk-jwt-version | 0.000 | 0.000 | 1.000 | correct | yes |
| pickle-allowed-modules | 0.000 | 0.000 | 1.000 | correct | yes |
| it-has-begun-first-half | 0.808 | 0.237 | 0.936 | incorrect | no |
| lootstash-strings-command | 0.000 | 0.000 | 1.000 | correct | yes |
| packedaway-compressor | 1.340 | 0.000 | 1.000 | correct | yes |
| flecks-file-type | 1.712 | 0.658 | 0.633 | incorrect | yes |
| randsubware-rounds | 1.118 | 1.118 | 0.486 | incorrect | no |
| frog-waf-field | 0.624 | 0.624 | 0.684 | correct | yes |
| eval-me-key | 0.368 | 0.368 | 0.918 | incorrect | no |
| diffecient-admin-criteria | 1.436 | 0.609 | 0.702 | incorrect | yes |
| back-to-past-hidden-dir | 1.008 | 0.690 | 0.540 | incorrect | no |
| motp-operator | 0.098 | 0.098 | 0.980 | incorrect | yes |

## Interpretation limits

This is a frequency-based black-box estimator, not a likelihood-weighted
reproduction of the original Semantic Entropy implementation.  The original
paper's formal self-consistent-error definition is also not applied because
there is no separate greedy T=0 response.  Semantic clusters were created
post-hoc by one annotator with access to reference answers, and the correctness
AUROC uses only 14 evaluable parent tasks.  These results support a pilot claim
and experimental design, not a calibrated deployment threshold.  Independent
cluster adjudication and a larger held-out task set are required for a paper's
confirmatory result.
