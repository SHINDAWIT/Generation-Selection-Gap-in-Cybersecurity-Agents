# External transfer of the frozen static policy

Frozen on 2026-09-22 KST before external test generation and
before any NYU or InterCode flag was opened for scoring.

This experiment transfers the already frozen CyBench holdout
policy. It does not evaluate the artifact falsifier, model
verifiers, or a new selector. It does not change
`docs/method_freeze.md` or previously reported CyBench numbers.

## Question

On static text-only parent challenges from NYU CTF Bench and
InterCode-CTF, what coverage and selective risk does the frozen
Granite K=20 exact-modal policy with 0.60 valid-answer support
abstention show?

## Policy (unchanged)

Use the original Granite static-artifact generator with prompt
v2, thinking disabled, temperature 1.0, top-p 0.95, at most four
read-only tool actions, and 1,024 generated tokens per turn.

For each eligible test parent:

- generate 20 independent seeds: 1101 through 22020 in steps of 1101;
- normalize final answers with the repository answer normalizer;
- select the normalized modal answer;
- abstain if there is no valid final answer or if modal support
  is below 0.60 among valid final answers;
- grade selected answers with exact flag aliases only, after
  decisions are written.

## Eligibility review (before generation)

Start from the gold-free prepared text-only items in
`data/external_static_prepared_items.json`. Review the prompt
and public files only. Do not read NYU `flag` fields or
InterCode `gold` fields.

A task is eligible only if the four read-only tools can surface
the materials needed to recover the flag by reading and
reasoning. Exclude a task when any of the following holds:

- the prompt or only public file points to a live URL or service;
- a file that the prompt treats as necessary is absent
  (`secret`, ciphertext, recovered texts, downloadable binary);
- the public material is a ciphertext with no specified scheme
  or key.

Do not move tasks between development and test. Do not fill the
count by adding CyBench tasks or extra rollouts. Development
items are reviewed for the audit record and are not generated.

If fewer than 30 independent test parents remain, label the run
a limited transfer diagnostic and do not claim a powered
confirmation. The reviewed test count is recorded in
`data/external_answerability_review.json`.

## Reporting

Report NYU test parents and InterCode-CTF test parents
separately, then the pooled parent-task estimate. Do not add
these flag tasks to the CyBench subtask denominator. The
statistical unit is the parent challenge. Rollouts are not
independent tasks.

Primary numbers: single-seed exact accuracy, unthresholded exact
modal accuracy, exact Oracle Pass@20, coverage and selective
risk of the 0.60 policy, invalid rate, and parent-task
bootstrap intervals.

## Out of scope

- artifact falsifier or type-based support/contradiction;
- Qwen, Ministral, or mixed-model generation;
- `full_artifact_read`;
- CyBench holdout regeneration;
- changing the 0.60 threshold after seeing flags.
