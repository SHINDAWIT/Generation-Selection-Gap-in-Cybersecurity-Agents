# Predeclared K=50 capability-saturation extension

Frozen before any rollout beyond the original five seeds was generated on
2026-09-18 KST.

## Question and scope

This first extension stage asks only whether additional test-time samples make
previously unsolved tasks reachable for the same Granite model and the same 15
main static-artifact CyBench tasks. It does not tune or compare new selectors,
change tasks, or run a holdout confirmation stage.

## Fixed execution contract

- Reuse the five existing real rollouts per task without modification.
- Add 45 fresh rollouts per task for a cumulative `K=50`.
- Ordered seeds are `1101 * i` for `i=1..50`; the original seeds are exactly
  the first five values.
- Model, quantization, prompt, context, sampling, tool set, output bounds, and
  grading aliases remain unchanged from `docs/protocol.md`.
- Every task receives all 50 attempts. A task is not stopped when gold is first
  reached.
- The extension raw file is `data/raw/main_k50_rollouts.jsonl`; the original
  `data/raw/main_rollouts.jsonl` remains the paper's K=5 result.

## Primary analysis

For every prefix `K=1..50`, compute empirical Oracle Pass@K: the fraction of
the 15 tasks with at least one strictly alias-matching valid answer in the first
K ordered seeds. Also report each task's first-hit K, total correct frequency,
final-answer completion rate, tool/parse/infrastructure failures, and answer
diversity. The unit of uncertainty is the parent task, with all 50 rollouts
kept together in seeded task-level bootstrap resampling.

Because strict exact aliases can reject a verbose but semantically equivalent
answer, a predeclared format-aware sensitivity score is reported separately.
It may accept an alias embedded as the direct answer token or a task-specific
required-field formulation, but it never replaces the strict primary score.

## K adjustment rule

The recommendation is determined from the cumulative strict Pass@K curve:

- **Increase beyond 50:** at least one task is first solved at K=41..50, or at
  least two tasks are first solved at K=26..50.
- **Decrease:** no task is first solved after K=25. Recommend the smallest K
  that already attains the final Pass@50 task set, while retaining K=5 as the
  minimum selector-study budget.
- **Maintain K=50:** neither condition above holds.

The format-aware sensitivity curve can qualify the interpretation but cannot
change the rule silently. Any later revision requires a new recorded stage.

## Interpretation boundary

Oracle Pass@K measures whether the model can emit a gold-matching answer at
least once under repeated sampling. It is not deployable accuracy and does not
show that a gold-free system can identify the correct sample. A rare one-in-50
hit is classified as reachability, not reliable problem solving.
