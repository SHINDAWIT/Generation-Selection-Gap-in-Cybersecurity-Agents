#!/usr/bin/env python3
"""Validate raw rollouts and produce selector metrics, CIs, tables, and SVG."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from statistics import mean
import sys
from typing import Any

from cisc_rollout.metrics import (
    bootstrap_percentile_ci,
    evaluate_tasks,
    paired_delta_stat,
    selector_stat,
)
from cisc_rollout.schema import Rollout, TaskItem, load_manifest, load_rollouts_jsonl


EXPECTED_SEEDS = (1101, 2202, 3303, 4404, 5505)
SELECTORS = (
    "single",
    "majority",
    "tool_grounded_majority",
    "agreement_abstention",
    "oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("pilot", "main"), required=True)
    parser.add_argument("--manifest", default="data/task_manifest.json")
    parser.add_argument("--raw")
    parser.add_argument("--check-pilot", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-stale-harness", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        record["_line_number"] = line_number
        records.append(record)
    return records


def validate_raw(
    tasks: list[TaskItem], records: list[dict[str, Any]], split: str
) -> dict[str, Any]:
    expected_tasks = {task.id for task in tasks}
    expected_pairs = {(task.id, seed) for task in tasks for seed in EXPECTED_SEEDS}
    seen: set[tuple[str, int]] = set()
    errors: list[str] = []
    for record in records:
        pair = (str(record.get("task_id")), int(record.get("seed", -1)))
        if pair in seen:
            errors.append(f"duplicate task/seed pair: {pair}")
        seen.add(pair)
        if record.get("split") != split:
            errors.append(f"wrong split at line {record['_line_number']}")
        if record.get("evidence_kind") != "real_model_benchmark":
            errors.append(f"non-real evidence at line {record['_line_number']}")
    if {task_id for task_id, _ in seen} - expected_tasks:
        errors.append("raw file contains undeclared task ids")
    missing = sorted(expected_pairs - seen)
    extra = sorted(seen - expected_pairs)
    if missing:
        errors.append(f"missing {len(missing)} task/seed pairs: {missing[:5]}")
    if extra:
        errors.append(f"unexpected task/seed pairs: {extra[:5]}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "task_count": len(tasks),
        "rollout_count": len(records),
        "expected_rollout_count": len(expected_pairs),
        "unique_task_seed_pairs": len(seen),
    }


def serializable_decisions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["per_task"]:
        rows.append(
            {
                **{key: value for key, value in row.items() if key != "selections"},
                "selections": {
                    name: asdict(selection) for name, selection in row["selections"].items()
                },
            }
        )
    return rows


def bootstrap_table(
    tasks: list[TaskItem],
    grouped: dict[str, list[Rollout]],
    samples: int,
    agreement_threshold: int = 3,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for selector in SELECTORS:
        output[selector] = {}
        for metric in ("overall_accuracy", "coverage", "selective_accuracy"):
            output[selector][metric] = bootstrap_percentile_ci(
                tasks,
                grouped,
                lambda summary, s=selector, m=metric: selector_stat(summary, s, m),
                samples=samples,
                seed=20260918,
                agreement_threshold=agreement_threshold,
            )
        if selector != "single":
            output[selector]["paired_delta_overall_vs_single"] = bootstrap_percentile_ci(
                tasks,
                grouped,
                paired_delta_stat(selector, "overall_accuracy", "single"),
                samples=samples,
                seed=20260918,
                agreement_threshold=agreement_threshold,
            )
    return output


def risk_coverage_points(
    decisions: list[dict[str, Any]], max_support: int = 5
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    total = len(decisions)
    for threshold in range(1, max_support + 1):
        answered = [
            row
            for row in decisions
            if not row["selections"]["majority"]["abstained"]
            and row["selections"]["majority"]["support"] >= threshold
        ]
        correct = sum(row["selections"]["majority"]["correct"] is True for row in answered)
        coverage = 0.0 if total == 0 else len(answered) / total
        accuracy = 0.0 if not answered else correct / len(answered)
        points.append(
            {
                "threshold": threshold,
                "answered": len(answered),
                "coverage": coverage,
                "selective_accuracy": accuracy,
                "selective_risk": 0.0 if not answered else 1.0 - accuracy,
            }
        )
    return points


def render_svg(
    points: list[dict[str, Any]],
    denominator: int = 5,
    label_thresholds: set[int] | None = None,
) -> str:
    width, height = 720, 520
    left, right, top, bottom = 90, 40, 40, 80
    plot_w, plot_h = width - left - right, height - top - bottom

    def xy(point: dict[str, Any]) -> tuple[float, float]:
        return (
            left + point["coverage"] * plot_w,
            top + point["selective_risk"] * plot_h,
        )

    ordered = sorted(points, key=lambda point: point["coverage"])
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(xy, ordered))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<polyline points="{polyline}" fill="none" stroke="#225ea8" stroke-width="3"/>',
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * plot_w
        y = top + value * plot_h
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#ddd"/>')
        elements.append(f'<text x="{x:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-size="14">{value:.1f}</text>')
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        elements.append(f'<text x="{left - 15}" y="{y + 5:.1f}" text-anchor="end" font-size="14">{value:.1f}</text>')
    for point in points:
        x, y = xy(point)
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#d7301f"/>')
        if label_thresholds is None or point["threshold"] in label_thresholds:
            elements.append(f'<text x="{x + 9:.1f}" y="{y - 9:.1f}" font-size="13">support≥{point["threshold"]}/{denominator}</text>')
    elements.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 22}" text-anchor="middle" font-size="17">Coverage</text>',
            f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="17">Selective risk</text>',
            f'<text x="{width / 2:.1f}" y="25" text-anchor="middle" font-size="19" font-weight="bold">Agreement risk–coverage curve</text>',
            '</svg>',
        ]
    )
    return "\n".join(elements)


def write_svg(
    points: list[dict[str, Any]],
    path: Path,
    denominator: int = 5,
    label_thresholds: set[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_svg(
            points,
            denominator=denominator,
            label_thresholds=label_thresholds,
        ),
        encoding="utf-8",
    )


def pilot_gate(records: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    total_turns = sum(len(record.get("turns", [])) for record in records)
    parse_failures = sum(int(record.get("parse_failure_count", 0)) for record in records)
    failed_infra_or_tool = sum(
        bool(record.get("infrastructure_failed")) or bool(record.get("tool_failed"))
        for record in records
    )
    projected_hours = mean(float(record.get("wall_seconds", 0.0)) for record in records) * 75 / 3600
    checks = {
        "five_items_x_five_rollouts": len(records) == 25,
        "model_turn_parse_rate_at_least_95pct": total_turns > 0 and (1 - parse_failures / total_turns) >= 0.95,
        "infrastructure_or_tool_failure_rate_at_most_20pct": failed_infra_or_tool / len(records) <= 0.20,
        "oracle_pass_at_5_nonzero": metrics["selectors"]["oracle"]["overall_accuracy"] > 0,
        "projected_main_runtime_at_most_6h": projected_hours <= 6.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "model_turn_count": total_turns,
        "parse_failure_count": parse_failures,
        "infrastructure_or_tool_failed_rollouts": failed_infra_or_tool,
        "projected_main_runtime_hours": projected_hours,
    }


def render_csv(metrics: dict[str, Any]) -> str:
    handle = io.StringIO(newline="")
    try:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "selector",
                "coverage",
                "selective_accuracy",
                "selective_risk",
                "overall_accuracy",
                "oracle_gap",
            ]
        )
        for selector in SELECTORS:
            row = metrics["selectors"][selector]
            writer.writerow(
                [
                    selector,
                    row["coverage"],
                    row["selective_accuracy"],
                    row["selective_risk"],
                    row["overall_accuracy"],
                    row["oracle_gap"],
                ]
            )
        return handle.getvalue()
    finally:
        handle.close()


def write_csv(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_csv(metrics), encoding="utf-8", newline="")


def render_paper_summary(metrics: dict[str, Any]) -> str:
    selectors = metrics["selectors"]
    majority = selectors["majority"]
    abstain = selectors["agreement_abstention"]
    grounded = selectors["tool_grounded_majority"]
    oracle = selectors["oracle"]
    single = selectors["single"]
    majority_delta = metrics["bootstrap"]["majority"]["paired_delta_overall_vs_single"]
    oracle_delta = metrics["bootstrap"]["oracle"]["paired_delta_overall_vs_single"]
    risk_by_threshold = {
        point["threshold"]: point for point in metrics["risk_coverage"]
    }
    return f"""# Paper-ready result summary

## Candidate title and claim

**When Agreement Misleads: Gold-Free Selection and Abstention for Local
Cybersecurity-Agent Rollouts.** On this exploratory static-artifact CyBench
subset, five-sample normalized majority improved task accuracy over one rollout
by {majority['overall_accuracy'] - single['overall_accuracy']:.3f}, but strong
agreement was often wrong. Agreement is useful as a coverage control, not as a
standalone correctness certificate.

## Experimental design

We evaluated five independent Granite 4.2 8B Q4_K_M rollouts on 15
predeclared static-artifact CyBench subtasks (75 real model runs). The agent had
read-only artifact tools and no network or arbitrary shell execution. The
statistical unit was the parent task; 95% percentile intervals use 2,000 seeded
task-level bootstrap resamples that keep all five within-task rollouts together.

## Main results

| Selector | Coverage | Selective accuracy | Overall accuracy | Oracle gap |
|---|---:|---:|---:|---:|
| Single | {single['coverage']:.3f} | {single['selective_accuracy']:.3f} | {single['overall_accuracy']:.3f} | {single['oracle_gap']:.3f} |
| Normalized majority | {majority['coverage']:.3f} | {majority['selective_accuracy']:.3f} | {majority['overall_accuracy']:.3f} | {majority['oracle_gap']:.3f} |
| Tool-grounded majority | {grounded['coverage']:.3f} | {grounded['selective_accuracy']:.3f} | {grounded['overall_accuracy']:.3f} | {grounded['oracle_gap']:.3f} |
| Agreement abstention (≥3/5) | {abstain['coverage']:.3f} | {abstain['selective_accuracy']:.3f} | {abstain['overall_accuracy']:.3f} | {abstain['oracle_gap']:.3f} |
| Oracle Pass@5 | {oracle['coverage']:.3f} | {oracle['selective_accuracy']:.3f} | {oracle['overall_accuracy']:.3f} | {oracle['oracle_gap']:.3f} |

Majority's paired overall-accuracy delta over Single had a bootstrap mean of
{majority_delta['mean']:.3f} (95% percentile interval
[{majority_delta['lower']:.3f}, {majority_delta['upper']:.3f}]); the interval
includes zero. Oracle Pass@5 exceeded Single by a bootstrap mean of
{oracle_delta['mean']:.3f} [{oracle_delta['lower']:.3f},
{oracle_delta['upper']:.3f}], showing limited headroom for any selector on this
task sample.

## Risk, abstention, and failure accounting

The ≥3/5 rule answered {abstain['coverage']:.3f} of tasks at
{abstain['selective_accuracy']:.3f} selective accuracy. Requiring ≥4/5 reduced
coverage to {risk_by_threshold[4]['coverage']:.3f} and selective risk to
{risk_by_threshold[4]['selective_risk']:.3f}; unanimity yielded
{risk_by_threshold[5]['coverage']:.3f} coverage and
{risk_by_threshold[5]['selective_risk']:.3f} risk. Yet the wrong-consensus rate
among ≥3/5 majority decisions was
{metrics['secondary']['wrong_consensus_rate']:.3f}, so agreement alone remains
unsafe. The evidence-grounded selector was also weak: stricter evidence
filtering lowered coverage without improving accuracy enough to beat majority.

Mean normalized answer diversity was
{metrics['secondary']['mean_diversity_ratio']:.3f}. Parse, tool, and
infrastructure failure rates were
{metrics['secondary']['parse_failure_rate']:.3f},
{metrics['secondary']['tool_failure_rate']:.3f}, and
{metrics['secondary']['infrastructure_failure_rate']:.3f}; the final-answer
completion rate was {metrics['secondary']['final_completion_rate']:.3f}. The 75
rollouts generated {metrics['secondary']['total_generated_tokens']} tokens in
{metrics['secondary']['total_wall_seconds']:.1f} aggregate wall-clock seconds.

## CISC paper interpretation

The estimand is the within-task difference among offline selectors applied to
the same five rollout logs; Oracle Pass@5 is a gold-only upper bound and not a
deployable method. The central result is negative but actionable: repeated
sampling creates some recoverable headroom, while naïve consensus and simple
quote grounding are insufficient safety mechanisms. A short paper should frame
the contribution as a reproducible failure analysis and risk-coverage baseline,
not a new state-of-the-art agent.

Public-task contamination, the small task sample, answer-string grading, one
model family, and the static read-only replacement for interactive CyBench
prevent claims about absolute cybersecurity capability or general deployment.
The tool-output bound was added after a diagnostic main attempt exposed context
overflow, so these replacement-run estimates are exploratory rather than
strictly confirmatory.

## Paper assets

Exact bootstrap intervals and per-threshold points are in
`results/main_metrics.json`; selector decisions are in
`data/processed/main_decisions.json`; the publication-ready risk–coverage plot
is `results/figures/risk_coverage.svg`; raw rollouts and non-fabrication hashes
are retained under `data/raw/` and `results/provenance.json`.
"""


def write_paper_summary(metrics: dict[str, Any]) -> None:
    Path("results/paper_ready_summary.md").write_text(
        render_paper_summary(metrics), encoding="utf-8"
    )


def build_provenance(raw_path: Path) -> dict[str, Any]:
    model_path = Path("models/granite-4.2-8b-Q4_K_M.gguf")
    runtime_archives = sorted(Path(".tools/downloads").glob("*b11026*.tar.gz"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_kind": "real_model_benchmark",
        "synthetic_rollouts_in_main": 0,
        "model_repository": "ibm-granite/granite-4.2-8b-GGUF",
        "model_file": str(model_path),
        "model_sha256": sha256_file(model_path),
        "runtime_release": "llama.cpp b11026",
        "runtime_archive_sha256": {
            path.name: sha256_file(path) for path in runtime_archives
        },
        "cybench_repository": "https://github.com/andyzorigin/cybench",
        "cybench_commit": "1097a7226eb034d3821208114da38f10b8627ab1",
        "protocol_sha256": sha256_file(Path("docs/protocol.md")),
        "manifest_sha256": sha256_file(Path("data/task_manifest.json")),
        "prepared_items_sha256": sha256_file(Path("data/prepared_items.json")),
        "raw_rollouts_sha256": sha256_file(raw_path),
        "tool_output_limits": {
            "per_result_chars": 4000,
            "cumulative_chars_per_rollout": 5000,
        },
        "harness_sha256": {
            str(path): sha256_file(path)
            for path in sorted(
                [*Path("src/cisc_rollout").glob("*.py")]
                + [
                    Path("scripts/run_experiment.py"),
                    Path("scripts/analyze_results.py"),
                    Path("scripts/prepare_cybench_subset.py"),
                ]
            )
        },
    }


def main() -> int:
    args = parse_args()
    raw_path = Path(args.raw or f"data/raw/{args.split}_rollouts.jsonl")
    tasks = [task for task in load_manifest(args.manifest) if task.split == args.split]
    if not raw_path.exists():
        raise SystemExit(f"raw rollout file is missing: {raw_path}")
    records = read_raw_records(raw_path)
    validation = validate_raw(tasks, records, args.split)
    grouped = load_rollouts_jsonl(str(raw_path))
    summary = evaluate_tasks(tasks, grouped)
    decisions = serializable_decisions(summary)
    risk_points = risk_coverage_points(decisions)
    core_metrics = {
        "evidence_kind": "real_model_benchmark",
        "split": args.split,
        "task_count": validation["task_count"],
        "rollout_count": validation["rollout_count"],
        "selectors": summary["selectors"],
        "secondary": {
            **summary["secondary"],
            "infrastructure_failure_rate": sum(
                bool(record.get("infrastructure_failed")) for record in records
            )
            / len(records),
            "final_completion_rate": sum(record.get("final") is not None for record in records)
            / len(records),
            "total_generated_tokens": sum(
                int(record.get("generated_tokens", 0)) for record in records
            ),
            "total_wall_seconds": sum(
                float(record.get("wall_seconds", 0.0)) for record in records
            ),
        },
        "risk_coverage": risk_points,
        "bootstrap": bootstrap_table(tasks, grouped, args.bootstrap_samples),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_paths = {
        "decisions": Path(f"data/processed/{args.split}_decisions.json"),
        "metrics": Path(f"results/{args.split}_metrics.json"),
        "csv": Path(f"results/{args.split}_metrics.csv"),
    }
    if args.validate_only:
        for path in output_paths.values():
            if not path.exists() or not path.stat().st_size:
                raise SystemExit(f"validation artifact is missing: {path}")
        stored = json.loads(output_paths["metrics"].read_text(encoding="utf-8"))
        for key in (
            "evidence_kind",
            "split",
            "task_count",
            "rollout_count",
            "selectors",
            "secondary",
            "risk_coverage",
            "bootstrap",
        ):
            if stored[key] != core_metrics[key]:
                raise SystemExit(f"stored metrics do not match raw recomputation: {key}")
        stored_decisions = json.loads(
            output_paths["decisions"].read_text(encoding="utf-8")
        )
        if stored_decisions != decisions:
            raise SystemExit("stored decisions do not match raw recomputation")
        if output_paths["csv"].read_bytes() != render_csv(core_metrics).encode("utf-8"):
            raise SystemExit("stored CSV does not match raw recomputation")
        if args.split == "main":
            svg_path = Path("results/figures/risk_coverage.svg")
            summary_path = Path("results/paper_ready_summary.md")
            provenance_path = Path("results/provenance.json")
            for path in (svg_path, summary_path, provenance_path):
                if not path.exists() or not path.stat().st_size:
                    raise SystemExit(f"validation artifact is missing: {path}")
            if svg_path.read_text(encoding="utf-8") != render_svg(risk_points):
                raise SystemExit("stored SVG does not match raw recomputation")
            if summary_path.read_text(encoding="utf-8") != render_paper_summary(core_metrics):
                raise SystemExit("stored paper summary does not match raw recomputation")
            stored_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            expected_provenance = build_provenance(raw_path)
            for key, value in expected_provenance.items():
                if key in {"generated_at", "harness_sha256"}:
                    continue
                if stored_provenance.get(key) != value:
                    raise SystemExit(f"stored provenance does not match current files: {key}")
            if (
                not args.allow_stale_harness
                and stored_provenance.get("harness_sha256")
                != expected_provenance["harness_sha256"]
            ):
                raise SystemExit("stored provenance does not match current files: harness_sha256")
        print(json.dumps(validation, indent=2))
        return 0

    output_paths["decisions"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["decisions"].write_text(json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
    output_paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["metrics"].write_text(json.dumps(core_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(core_metrics, output_paths["csv"])

    gate = None
    if args.split == "pilot" and args.check_pilot:
        gate = pilot_gate(records, core_metrics)
        Path("results/pilot_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print(json.dumps(gate, indent=2))
        if not gate["passed"]:
            return 2
    if args.split == "main":
        write_svg(risk_points, Path("results/figures/risk_coverage.svg"))
        provenance = build_provenance(raw_path)
        Path("results/provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        write_paper_summary(core_metrics)
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
