# Method freeze for external static transfer

Frozen on 2026-09-22 KST after the blinded answerability review
and before any external test generation or flag scoring.

This freeze does not replace `docs/method_freeze.md`. The CyBench
holdout policy stays byte-for-byte as written there.

## Policy copied from the CyBench freeze

Use the original Granite static-artifact generator with prompt v2,
thinking disabled, temperature 1.0, top-p 0.95, at most four tool
actions, and 1,024 generated tokens per turn.

For each eligible external test parent:

- generate 20 independent seeds: 1101 through 22020 in steps of 1101;
- normalize final answers with the repository answer normalizer;
- select the normalized modal answer;
- abstain if there is no valid final answer or if the modal support
  is below 0.60 among valid final answers;
- grade selected answers with exact flag aliases only, after
  decisions are written.

## Cohort

Generate only the 35 test parents marked `generated: true` in
`data/external_answerability_review.json` and listed in
`data/external_transfer_prepared_items.json`.

- NYU CTF Bench test: 7
- InterCode-CTF test: 28
- Development items: reviewed, not generated
- Artifact falsifier: not used

The reviewed test count is 35, so a transfer attempt is allowed.
It remains a static text-only flag transfer, not a CyBench
subtask result and not a powered general-method claim.

## Input hashes at freeze time

- `docs/method_freeze.md`:
  `c6fbe2b0c9aeb9b3697ea79dcb455917ec07d8cd7558672bcdf1e3eab27b81a2`
- `docs/external_policy_transfer_protocol.md`:
  `3e7b4d4a0ea79940e7a92a0c955821129da5fe37f811fe28ab362f4411774ba7`
- `data/external_answerability_review.json`:
  `1a69c0b3be62752166936594ae3d0f681734a7d7dc9c65f0954406083e13a927`
- `data/external_transfer_prepared_items.json`:
  `e9fe15efd99eb23ef0ac37a8a0975c8c8f5f53f2bbfb5262f327430f37f6e2f2`
- `data/external_static_prepared_items.json`:
  `8ac47bccf3f604e55dff6ea70499d091dd1363296e8e9d76c962a74efb1e903f`

## Not used

Quote-presence verifiers, Granite/Qwen model verifiers, the
artifact falsifier, mixed-model generation, and CyBench holdout
tasks.
