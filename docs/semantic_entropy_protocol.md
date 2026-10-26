# Retrospective K=50 semantic-entropy analysis protocol

Frozen on 2026-09-20 KST before the first entropy metrics file was produced.
The K=50 answers had already been inspected while designing the clustering
rules, so this is a retrospective, hypothesis-generating analysis rather than
a preregistered confirmation study.

## Research question

For the existing 15 static-artifact CyBench tasks, does agreement measured over
exact answer strings or semantic answer clusters distinguish correct from
incorrect modal answers?  The analysis also asks whether low-entropy wrong
consensus occurs despite the presence of a correct minority candidate.

## Fixed evidence

- Source: `data/raw/main_k50_rollouts.jsonl` (15 tasks x 50 ordered seeds).
- No new model generations are used.
- The parent task is the statistical unit.  The 750 rollouts are never treated
  as 750 independent evaluation examples.
- Prefix checkpoints are K in {5, 10, 15, 25, 50}; K=50 is primary.

## Entropy estimators

For observed cluster counts n_c among N samples, the empirical entropy in nats
is

    H = - sum_c (n_c / N) log(n_c / N).

Two clusterings are reported:

1. Exact-answer entropy uses the repository's conservative answer
   normalization and treats each distinct normalized string as a cluster.
2. Empirical semantic entropy applies the task-specific, auditable rules in
   `data/annotations/k50_semantic_clusters.json`.

The primary entropy conditions on valid parsed final answers.  A sensitivity
analysis adds all missing or parse-failed finals as one `__no_answer__` cluster.
Normalized entropy divides by log(N), with zero returned for N <= 1.

This is a black-box frequency estimator.  It is not claimed to exactly
reproduce likelihood-weighted semantic entropy because the archived rollout
records contain token counts but not per-token log-probabilities.

## Semantic clustering and correctness

Clustering is deterministic and task-local.  Each observed valid normalized
answer must match exactly one regex rule.  Rules preserve security-significant
distinctions such as `==` versus `!==`, numeric round counts, version strings,
paths, and flags.  Natural-language paraphrases such as descriptions of RSA or
UPX may collapse into one semantic cluster.

Each cluster is annotated `correct`, `incorrect`, or `ambiguous`.  Correctness
was not annotated blind to the task reference answers.  `ambiguous` modal tasks
are excluded from correctness discrimination metrics but remain in entropy and
coverage summaries.  This single-annotator semantic audit must be independently
adjudicated before confirmatory publication claims.

## Primary outcomes

- Per-task exact and semantic entropy at K=50.
- Modal semantic support and modal-cluster correctness.
- AUROC for detecting correct modal answers using negative entropy as
  confidence; ties receive half credit.
- Task-stratified bootstrap intervals for AUROC.
- Grouped risk-coverage points and AURC, accepting lower-entropy tasks first.
- Counts of semantically unanimous wrong tasks and wrong modal clusters with at
  least 0.8 or 0.9 support among valid answers.
- Semantic Oracle Pass@K: at least one correct semantic cluster appears.

Because there are only 15 parent tasks, all AUROC, confidence intervals, and
risk-coverage results are exploratory.  No threshold is tuned and tested on
this same set for a deployment claim.

## Interpretation boundary

The original *Too Consistent to Detect* definition also requires a wrong greedy
response and semantic equivalence between that greedy response and every
stochastic sample.  This dataset has no separately generated temperature-zero
greedy response and uses temperature 1.0/top-p 0.95.  Therefore results are
described as semantic unanimity, low-entropy wrong consensus, or near-unanimous
wrong consensus, not as formal self-consistent errors under that paper's
definition.

## Validation and stop condition

The analysis is complete when all 750 records and the 15 x 50 seed grid are
validated, every observed valid normalized answer matches exactly one semantic
cluster, deterministic unit tests pass, generated artifacts reproduce under
`--validate-only`, and the project critic check passes.
