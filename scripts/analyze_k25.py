#!/usr/bin/env python3
"""Run and validate the complete K=25 selector/abstention analysis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from cisc_rollout.metrics import evaluate_tasks
from cisc_rollout.schema import load_manifest, load_rollouts_jsonl
from analyze_results import (
    SELECTORS,
    bootstrap_table,
    read_raw_records,
    render_csv,
    render_svg,
    risk_coverage_points,
    serializable_decisions,
    sha256_file,
)


SEEDS = tuple(1101 * index for index in range(1, 26))
K = 25
AGREEMENT_THRESHOLD = 15
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260918
SVG_LABELS = {1, 5, 10, 15, 20, 25}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/main_k25_rollouts.jsonl")
    parser.add_argument("--source", default="data/raw/main_k50_rollouts.jsonl")
    parser.add_argument("--baseline", default="data/raw/main_rollouts.jsonl")
    parser.add_argument("--manifest", default="data/task_manifest.json")
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-stale-harness", action="store_true")
    return parser.parse_args()


def validate_raw(
    records: list[dict[str, Any]], task_ids: list[str], source_path: Path
) -> dict[str, Any]:
    expected = {(task_id, seed) for task_id in task_ids for seed in SEEDS}
    pairs: list[tuple[str, int]] = []
    errors: list[str] = []
    for record in records:
        pair = (str(record.get("task_id")), int(record.get("seed", -1)))
        pairs.append(pair)
        if record.get("split") != "main":
            errors.append(f"wrong split: {pair}")
        if record.get("evidence_kind") != "real_model_benchmark":
            errors.append(f"non-real evidence: {pair}")
        if record.get("model") != "granite-4.2-8b-Q4_K_M":
            errors.append(f"wrong model: {pair}")
        if record.get("temperature") != 1.0 or record.get("top_p") != 0.95:
            errors.append(f"wrong sampling settings: {pair}")
        if bool(record.get("infrastructure_failed")):
            errors.append(f"infrastructure failure: {pair}")
    if len(pairs) != len(set(pairs)):
        errors.append("duplicate task/seed pair")
    if set(pairs) != expected:
        errors.append(
            f"K=25 grid mismatch: missing={len(expected - set(pairs))}, "
            f"extra={len(set(pairs) - expected)}"
        )

    source_records = {
        (str(record["task_id"]), int(record["seed"])): record
        for record in read_raw_records(source_path)
        if int(record.get("seed", -1)) in SEEDS
    }
    for record in records:
        clean_record = {key: value for key, value in record.items() if key != "_line_number"}
        pair = (str(record["task_id"]), int(record["seed"]))
        source_record = source_records.get(pair)
        if source_record is not None:
            source_record = {
                key: value for key, value in source_record.items() if key != "_line_number"
            }
        if clean_record != source_record:
            errors.append(f"record differs from K=50 source: {pair}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "task_count": len(task_ids),
        "rollout_count": len(records),
        "rollouts_per_task": K,
        "unique_task_seed_pairs": len(set(pairs)),
    }


def build_provenance(
    raw_path: Path, source_path: Path, baseline_path: Path
) -> dict[str, Any]:
    model_path = Path("models/granite-4.2-8b-Q4_K_M.gguf")
    runtime_archives = sorted(Path(".tools/downloads").glob("*b11026*.tar.gz"))
    tracked_code = sorted(
        [*Path("src/cisc_rollout").glob("*.py")]
        + [
            Path("scripts/run_experiment.py"),
            Path("scripts/prepare_k25_evidence.py"),
            Path("scripts/analyze_k25.py"),
            Path("scripts/prepare_cybench_subset.py"),
        ]
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "real_model_benchmark",
        "analysis_status": "post_hoc_after_k50_inspection",
        "synthetic_rollouts": 0,
        "k": K,
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "model_repository": "ibm-granite/granite-4.2-8b-GGUF",
        "model_file": str(model_path),
        "model_sha256": sha256_file(model_path),
        "runtime_release": "llama.cpp b11026",
        "runtime_archive_sha256": {
            path.name: sha256_file(path) for path in runtime_archives
        },
        "cybench_commit": "1097a7226eb034d3821208114da38f10b8627ab1",
        "protocol_sha256": sha256_file(Path("docs/k25_protocol.md")),
        "manifest_sha256": sha256_file(Path("data/task_manifest.json")),
        "prepared_items_sha256": sha256_file(Path("data/prepared_items.json")),
        "raw_file": str(raw_path),
        "raw_sha256": sha256_file(raw_path),
        "source_k50_file": str(source_path),
        "source_k50_sha256": sha256_file(source_path),
        "baseline_k5_file": str(baseline_path),
        "baseline_k5_sha256": sha256_file(baseline_path),
        "harness_sha256": {str(path): sha256_file(path) for path in tracked_code},
    }


def render_paper_summary(metrics: dict[str, Any]) -> str:
    selectors = metrics["selectors"]
    single = selectors["single"]
    majority = selectors["majority"]
    grounded = selectors["tool_grounded_majority"]
    abstain = selectors["agreement_abstention"]
    oracle = selectors["oracle"]
    majority_delta = metrics["bootstrap"]["majority"][
        "paired_delta_overall_vs_single"
    ]
    oracle_delta = metrics["bootstrap"]["oracle"][
        "paired_delta_overall_vs_single"
    ]
    risks = {row["threshold"]: row for row in metrics["risk_coverage"]}
    secondary = metrics["secondary"]
    return f"""# K=25 paper-ready result summary

## Status and design

This exploratory post-hoc extension applies the complete selector and
abstention pipeline to 25 ordered Granite 4.2 8B Q4_K_M rollouts for each of 15
static-artifact CyBench tasks (375 real model records). It reuses the exact
K=25 prefix of the previously acquired K=50 evidence. The parent task is the
statistical unit; intervals use {metrics['bootstrap_samples']:,} seeded
task-level bootstrap resamples that keep all 25 rollouts together.

## Main results

| Selector | Coverage | Selective accuracy | Overall accuracy | Oracle gap |
|---|---:|---:|---:|---:|
| Single | {single['coverage']:.3f} | {single['selective_accuracy']:.3f} | {single['overall_accuracy']:.3f} | {single['oracle_gap']:.3f} |
| Normalized majority | {majority['coverage']:.3f} | {majority['selective_accuracy']:.3f} | {majority['overall_accuracy']:.3f} | {majority['oracle_gap']:.3f} |
| Tool-grounded majority | {grounded['coverage']:.3f} | {grounded['selective_accuracy']:.3f} | {grounded['overall_accuracy']:.3f} | {grounded['oracle_gap']:.3f} |
| Agreement abstention (≥15/25) | {abstain['coverage']:.3f} | {abstain['selective_accuracy']:.3f} | {abstain['overall_accuracy']:.3f} | {abstain['oracle_gap']:.3f} |
| Oracle Pass@25 | {oracle['coverage']:.3f} | {oracle['selective_accuracy']:.3f} | {oracle['overall_accuracy']:.3f} | {oracle['oracle_gap']:.3f} |

Normalized majority changed overall accuracy by
{majority['overall_accuracy'] - single['overall_accuracy']:+.3f} versus Single.
Its paired bootstrap delta mean was {majority_delta['mean']:+.3f} with a 95%
percentile interval [{majority_delta['lower']:+.3f},
{majority_delta['upper']:+.3f}]. Oracle Pass@25 exceeded Single by a bootstrap
mean of {oracle_delta['mean']:+.3f} [{oracle_delta['lower']:+.3f},
{oracle_delta['upper']:+.3f}].

## Abstention and risk

The fixed ≥15/25 agreement rule covered {abstain['coverage']:.3f} of tasks at
{abstain['selective_accuracy']:.3f} selective accuracy. Descriptive support
thresholds 10/25, 15/25, 20/25, and 25/25 yielded coverage/risk pairs of
({risks[10]['coverage']:.3f}, {risks[10]['selective_risk']:.3f}),
({risks[15]['coverage']:.3f}, {risks[15]['selective_risk']:.3f}),
({risks[20]['coverage']:.3f}, {risks[20]['selective_risk']:.3f}), and
({risks[25]['coverage']:.3f}, {risks[25]['selective_risk']:.3f}), respectively.
The wrong-consensus rate among ≥15/25 majority decisions was
{secondary['wrong_consensus_rate']:.3f}.

## Failure accounting and interpretation

Final-answer completion, parse-failure, tool-failure, and infrastructure-
failure rates were {secondary['final_completion_rate']:.3f},
{secondary['parse_failure_rate']:.3f}, {secondary['tool_failure_rate']:.3f}, and
{secondary['infrastructure_failure_rate']:.3f}. The 375 records generated
{secondary['total_generated_tokens']} tokens in
{secondary['total_wall_seconds']:.1f} aggregate wall-clock seconds.

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
"""


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw)
    source_path = Path(args.source)
    baseline_path = Path(args.baseline)
    tasks = [task for task in load_manifest(args.manifest) if task.split == "main"]
    records = read_raw_records(raw_path)
    validation = validate_raw(records, [task.id for task in tasks], source_path)
    grouped = load_rollouts_jsonl(str(raw_path))
    summary = evaluate_tasks(
        tasks, grouped, agreement_threshold=AGREEMENT_THRESHOLD
    )
    decisions = serializable_decisions(summary)
    risk_points = risk_coverage_points(decisions, max_support=K)
    metrics = {
        "schema_version": 1,
        "evidence_kind": "real_model_benchmark",
        "analysis_status": "post_hoc_after_k50_inspection",
        "split": "main",
        "k": K,
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "task_count": validation["task_count"],
        "rollout_count": validation["rollout_count"],
        "selectors": summary["selectors"],
        "secondary": {
            **summary["secondary"],
            "infrastructure_failure_rate": sum(
                bool(record.get("infrastructure_failed")) for record in records
            )
            / len(records),
            "final_completion_rate": sum(
                record.get("final") is not None for record in records
            )
            / len(records),
            "total_generated_tokens": sum(
                int(record.get("generated_tokens", 0)) for record in records
            ),
            "total_wall_seconds": sum(
                float(record.get("wall_seconds", 0.0)) for record in records
            ),
        },
        "risk_coverage": risk_points,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap": bootstrap_table(
            tasks,
            grouped,
            args.bootstrap_samples,
            agreement_threshold=AGREEMENT_THRESHOLD,
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    decisions_path = Path("data/processed/k25_decisions.json")
    metrics_path = Path("results/k25_metrics.json")
    csv_path = Path("results/k25_metrics.csv")
    summary_path = Path("results/k25_paper_ready_summary.md")
    svg_path = Path("results/figures/k25_risk_coverage.svg")
    provenance_path = Path("results/k25_provenance.json")
    expected_svg = render_svg(
        risk_points, denominator=K, label_thresholds=SVG_LABELS
    )
    expected_summary = render_paper_summary(metrics)

    if args.validate_only:
        stored_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for key, value in metrics.items():
            if key != "generated_at" and stored_metrics.get(key) != value:
                raise SystemExit(f"stored K=25 metrics mismatch: {key}")
        if json.loads(decisions_path.read_text(encoding="utf-8")) != decisions:
            raise SystemExit("stored K=25 decisions mismatch")
        if csv_path.read_bytes() != render_csv(metrics).encode("utf-8"):
            raise SystemExit("stored K=25 CSV mismatch")
        if summary_path.read_text(encoding="utf-8") != expected_summary:
            raise SystemExit("stored K=25 paper summary mismatch")
        if svg_path.read_text(encoding="utf-8") != expected_svg:
            raise SystemExit("stored K=25 SVG mismatch")
        stored_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_provenance = build_provenance(raw_path, source_path, baseline_path)
        for key, value in expected_provenance.items():
            if key in {"generated_at", "harness_sha256"}:
                continue
            if stored_provenance.get(key) != value:
                raise SystemExit(f"stored K=25 provenance mismatch: {key}")
        if (
            not args.allow_stale_harness
            and stored_provenance.get("harness_sha256")
            != expected_provenance["harness_sha256"]
        ):
            raise SystemExit("stored K=25 provenance mismatch: harness_sha256")
        print(json.dumps(validation, indent=2))
        return 0

    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    decisions_path.write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path.write_text(render_csv(metrics), encoding="utf-8", newline="")
    summary_path.write_text(expected_summary, encoding="utf-8")
    svg_path.write_text(expected_svg, encoding="utf-8")
    provenance_path.write_text(
        json.dumps(
            build_provenance(raw_path, source_path, baseline_path), indent=2
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "validation": validation,
                "selectors": metrics["selectors"],
                "wrong_consensus_rate": metrics["secondary"][
                    "wrong_consensus_rate"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
