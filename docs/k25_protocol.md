# K=25 end-to-end selector study protocol

Frozen on 2026-09-18 KST before computing any K=25 selector, abstention, or
risk-coverage result. This stage was requested after the K=50 capability curve
had already been inspected, so it is a transparent post-hoc extension rather
than a prospective confirmation experiment.

## Question and scope

Apply the complete Topic-1 pipeline at a cumulative budget of 25 independent
rollouts per task: raw-evidence validation, gold-free selection, abstention,
Oracle reachability, task-level uncertainty, failure accounting, provenance,
paper-ready interpretation, and completion audit.

The frozen 15-task main static-artifact CyBench subset is unchanged. The first
25 ordered seeds from `data/raw/main_k50_rollouts.jsonl` are reused exactly;
regenerating identical model calls would add cost without adding independent
evidence. Pilot and harness gates were already passed before the original main
study and are inherited unchanged.

## Fixed data contract

- `K=25` for every main task, for 375 real-model records in total.
- Ordered seeds: `1101 * i` for `i=1..25`.
- Model, quantization, prompt, sampling, tools, context and output bounds,
  normalization, task manifest, and strict gold aliases remain unchanged.
- No outcome-based early stopping or task-specific rollout allocation.
- The first five records for every task must equal the original K=5 evidence,
  and all 375 records must equal the corresponding K=50 source records.
- Synthetic or fixture records are excluded.

## Fixed selectors

1. **Single:** first valid rollout in ordered-seed order.
2. **Normalized majority:** most frequent exact normalized answer among valid
   rollouts; earliest seed breaks a tie.
3. **Tool-grounded majority:** the same rule after retaining only answers with
   a verified evidence quote in a successful tool observation.
4. **Agreement abstention:** return normalized majority only when its absolute
   support is at least `15/25`; otherwise abstain. This preserves the original
   nominal `3/5 = 60%` threshold. Parse failures do not lower the threshold.
5. **Oracle Pass@25:** gold-only indicator that at least one valid rollout
   strictly matches a registered alias. It is not deployable.

The primary comparison is paired overall accuracy against Single. Coverage,
selective accuracy/risk, and Oracle gap are reported for every selector.
Wrong-consensus rate uses the same primary `15/25` support threshold.

## Risk-coverage and uncertainty

- Sweep absolute majority-support thresholds `1..25` without choosing a new
  threshold from their outcomes.
- Highlight `10/25`, `15/25`, `20/25`, and `25/25` only as descriptive points.
- Use 2,000 seeded nonparametric bootstrap resamples of the 15 parent tasks,
  keeping all 25 within-task rollouts together.
- Report parse, tool, infrastructure, final-completion, answer-diversity,
  generated-token, and aggregate wall-time statistics.

## Interpretation boundary

This is an end-to-end reuse analysis of already acquired evidence. Because the
K=50 reachability curve and grading sensitivity were seen before this protocol,
K=25 results may support an exploratory CISC analysis but are not a clean
holdout confirmation. Exact-string majority can fragment semantically similar
answers; that limitation must be discussed rather than repaired after seeing
selector outcomes.
