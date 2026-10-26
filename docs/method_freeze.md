# Method freeze for untouched static holdout

Frozen on 2026-09-20 KST after Stage A and Stage B development analyses, and
before materializing holdout gold answers.

## Frozen deployable policy

Use the original Granite static-artifact generator with prompt v2, thinking
disabled, temperature 1.0, top-p 0.95, at most four tool actions, and 1,024
generated tokens per turn.

For each holdout task:

- generate 20 independent seeds: 1101 through 22020 in steps of 1101;
- normalize final answers with the repository answer normalizer;
- select the normalized modal answer;
- abstain if there is no valid final answer or if the modal support is below
  0.60 among valid final answers;
- grade selected answers with exact gold aliases only.

## What is not frozen as a deployable policy

The six-task deterministic diagnostic selector is not used on holdout.  It is a
task-specific upper-bound diagnostic showing that stronger validators can
intercept wrong consensus on known development tasks.

The cross-model mixed Granite/Qwen condition is not expanded to holdout because
the predeclared Stage B diversity gate failed: it did not improve oracle recall
and increased incorrect semantic modals on the six diagnostic tasks.

Quote presence alone is not used as an entailment verifier.

## Holdout reporting

Report exact majority accuracy, exact oracle pass@20, selective risk at the
0.60 support threshold, invalid rate, and average rollout cost.  Treat this as
a low-power static CyBench confirmation, not a full interactive CyBench claim.
