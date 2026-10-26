# Generation–Selection Gap in Cybersecurity Agents

**Separating correct-candidate generation, modal selection, and abstention in a local cybersecurity agent.**

[Paper PDF](CISC-W_2026_paper__324.pdf) · [한국어](README.md) · [Code guide](docs/code-guide.md) · [Methods and limitations](docs/methodology.md) · [Reproducibility](docs/reproducibility.md)

## Research question

Repeated sampling can produce a correct answer without returning it to the user. This study examines where correct answers are lost between **candidate generation, normalized-string modal selection, and abstention**. It uses an existing local model and inference engine; it does not introduce a newly trained model.

## Main findings

On 15 static CyBench development subtasks, increasing K from 5 to 25 raised strict gold-alias Oracle success from **4/15 to 7/15**, while modal correctness rose from **3/15 to 4/15**. The additional reachable answers were only partly recovered by modal selection.

![Generation versus selection, and outcomes after abstention](docs/assets/generation-selection-gap.png)

| Evaluation | Result | Interpretation |
| --- | --- | --- |
| Development, 15 parents, K=5 → 25 | Oracle 4 → 7; correct modal 3 → 4 | Candidate availability and returned-answer correctness differ |
| MOTP, K=50 | Wrong `!==` 49 times; correct `==` once | Near-unanimity can hide a correct minority answer |
| Unused CyBench, 11 parents, K=20 | 6 correct, 2 wrong, 3 abstentions | The frozen threshold preserved 6 correct answers and intercepted 3 wrong modals |
| External static transfer, 35 parents, K=20 | 2 correct, 7 wrong, 26 abstentions; Oracle 3/35 | Candidate absence and rejection of a correct answer also limit performance |

Oracle uses reference answers offline and is not a deployable selector. The public figure was created from preserved counts; its two panels use different cohorts and generation settings.

## What the repository demonstrates

- A repeated-evaluation runner with independent seed histories, bounded read-only tools, and structured logs.
- A local llama.cpp HTTP client, answer normalization, parsing, modal selection, and explicit abstention.
- Failure accounting that separates candidate absence, selection loss, invalid responses, and correct-answer rejection.
- A public ledger of **1,670 projected rollout records**, code snapshots, tests, and file-integrity metadata.

The paper has eight coauthors. The publication commit does not establish individual ownership of every function. See [authors](CITATION.md) and [provenance](docs/provenance.md).

## Verify without a GPU

Python **3.11+**, standard library only:

```bash
git clone https://github.com/SHINDAWIT/Generation-Selection-Gap-in-Cybersecurity-Agents.git
cd Generation-Selection-Gap-in-Cybersecurity-Agents
python3 -B scripts/verify_public_results.py
PYTHONPATH=src:. python3 -B -m unittest discover -s tests -v
```

The verifier recounts decisions from normalized-answer SHA-256 keys and inherited scoring labels. It checks full task/seed grids and published file hashes. It does **not** independently regrade original answer text, regenerate semantic annotations, rerun bootstrap intervals, or run inference. Full model conversations, tool outputs, and external flag answer files are not redistributed.

## Scope

The study uses **one quantized model**, IBM Granite 4.2 8B Q4_K_M, and **static read-only tasks**. There are 15 development, 11 unused CyBench, and 35 external parent tasks. The development K=25 analysis reuses the K=50 prefix. Retrospective semantic labels came from one annotator with access to reference answers and are separate from strict primary grading.

The current runner uses prompt v2/v3, while the historical development logs use v1. Historical K=25 source hashes also differ from the current snapshot. The repository documents these differences rather than asserting exact historical regeneration. [Methods](docs/methodology.md) · [Version audit](docs/provenance.md)

Paper: **The Gap Between Correct-Answer Generation and Modal Answer Selection in Multi-Rollout Cybersecurity Agents**.

[Detailed counts](docs/results.md) · [Upstream sources](THIRD_PARTY_NOTICES.md)
