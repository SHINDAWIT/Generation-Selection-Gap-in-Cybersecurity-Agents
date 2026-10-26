#!/usr/bin/env python3
"""Analyze the frozen-policy external transfer. Score only after gold exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.analyze_holdout import (
    SEEDS_20,
    SUPPORT_THRESHOLD,
    analyze_task,
    bootstrap_ci,
    load_json,
    load_jsonl,
    require_grid,
    selector_summaries,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared", default="data/external_transfer_prepared_items.json"
    )
    parser.add_argument("--raw", default="data/raw/external_transfer_k20_rollouts.jsonl")
    parser.add_argument("--gold", default="data/external_transfer_gold.json")
    parser.add_argument("--decisions", default="results/external_transfer_decisions.json")
    parser.add_argument("--output", default="results/external_transfer_metrics.json")
    parser.add_argument("--summary", default="results/external_transfer_summary.md")
    parser.add_argument(
        "--provenance", default="results/external_transfer_provenance.json"
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--decisions-only", action="store_true")
    return parser.parse_args()


def attach_gold(items: list[dict[str, Any]], gold_path: Path) -> list[dict[str, Any]]:
    if not gold_path.is_file():
        return [{**item, "gold_aliases": []} for item in items]
    gold = load_json(gold_path)
    by_id = {str(row["id"]): row for row in gold.get("items", [])}
    merged: list[dict[str, Any]] = []
    for item in items:
        extra = by_id.get(str(item["id"]))
        if extra is None:
            raise ValueError(f"gold file missing task: {item['id']}")
        merged.append({**item, "gold_aliases": list(extra.get("gold_aliases", []))})
    return merged


def source_groups(task_rows: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    source_by_id = {str(item["id"]): str(item["source"]) for item in items}
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in task_rows:
        source = source_by_id[str(row["task_id"])]
        groups.setdefault(source, []).append(row)
    return groups


def _short_source(name: str) -> str:
    if name == "NYU CTF Bench":
        return "NYU"
    if name == "InterCode-CTF":
        return "InterCode"
    return name


def _policy_line(block: dict[str, Any], task_count: int, scored: bool) -> list[str]:
    policy = block["exact_majority_support_ge_0_60"]
    lines = [
        f"Policy answered {policy['answered']}/{task_count}, "
        f"coverage {policy['coverage']:.1%}."
    ]
    if scored:
        lines.extend(
            [
                f"Correct {policy['correct']}, incorrect {policy['incorrect']}, "
                f"abstained {policy['abstained']}.",
                f"Selective risk {policy['selective_risk']:.1%}.",
                f"Single seed {block['single_seed_1101']['correct']}/{task_count}.",
                f"Exact modal "
                f"{block['exact_modal_without_support_abstention']['correct']}/"
                f"{task_count}.",
                f"Oracle Pass@20 {block['exact_oracle_pass_at_20']['correct']}/"
                f"{task_count}.",
            ]
        )
    return lines


def _task_table(analysis: dict[str, Any], items: list[dict[str, Any]], scored: bool) -> list[str]:
    source_by_id = {str(item["id"]): str(item["source"]) for item in items}
    header = (
        "| Task | Source | Support | Modal correct | Selected | Oracle | Invalid |"
        if scored
        else "| Task | Source | Support | Selected | Invalid |"
    )
    align = (
        "| --- | --- | ---: | --- | --- | --- | ---: |"
        if scored
        else "| --- | --- | ---: | --- | ---: |"
    )
    lines = [header, align]
    for row in analysis["tasks"]:
        selected = "abstain" if row["abstained"] else str(row.get("selected_correct"))
        source = _short_source(source_by_id[str(row["task_id"])])
        if scored:
            lines.append(
                f"| {row['task_id']} | {source} | {row['exact_mode_support']:.2f} | "
                f"{row['exact_mode_correct']} | {selected} | {row['oracle_correct']} | "
                f"{row['invalid_count']} |"
            )
        else:
            lines.append(
                f"| {row['task_id']} | {source} | {row['exact_mode_support']:.2f} | "
                f"{'abstain' if row['abstained'] else 'answer'} | {row['invalid_count']} |"
            )
    return lines


def render_summary(
    analysis: dict[str, Any],
    scored: bool,
    items: list[dict[str, Any]] | None = None,
) -> str:
    overall = analysis["selectors"]
    invalid = sum(int(row["invalid_count"]) for row in analysis["tasks"])
    rollouts = analysis["task_count"] * 20
    lines = [
        "# External transfer of the frozen K=20 policy",
        "",
        "This is not a CyBench subtask result and does not evaluate the",
        (
            "artifact falsifier. Decisions were written before flags were used."
            if scored
            else "artifact falsifier. This file is decisions-only; flags are not opened."
        ),
        "",
        f"Test parents: {analysis['task_count']}.",
        f"NYU: {analysis['source_counts'].get('NYU CTF Bench', 0)}.",
        f"InterCode-CTF: {analysis['source_counts'].get('InterCode-CTF', 0)}.",
        f"Invalid rollouts: {invalid}/{rollouts} ({invalid / rollouts:.1%}).",
        "",
        *_policy_line(overall, analysis["task_count"], scored),
    ]
    if scored:
        effect = overall["abstention_effect"]
        bootstrap = analysis["bootstrap"]
        lines.extend(
            [
                f"Abstention intercepted {effect['wrong_modal_intercepted']} wrong "
                f"modals and {effect['correct_modal_abstained']} correct modal"
                f"{'' if effect['correct_modal_abstained'] == 1 else 's'}; "
                f"descriptive risk reduction {effect['risk_reduction']:.3f}.",
                "Selected overall accuracy bootstrap 95% CI: "
                f"[{bootstrap['selected_overall_accuracy']['lower']:.3f}, "
                f"{bootstrap['selected_overall_accuracy']['upper']:.3f}].",
                "Selective risk bootstrap 95% CI: "
                f"[{bootstrap['selective_risk']['lower']:.3f}, "
                f"{bootstrap['selective_risk']['upper']:.3f}].",
                "Abstention risk-reduction bootstrap 95% CI: "
                f"[{bootstrap['abstention_risk_reduction']['lower']:.3f}, "
                f"{bootstrap['abstention_risk_reduction']['upper']:.3f}].",
            ]
        )
    else:
        lines.append("Correctness is not reported in the decisions-only file.")
    lines.append("")
    for source, block in analysis["by_source"].items():
        lines.append(f"## {source}")
        lines.extend(_policy_line(block, block["task_count"], scored))
        lines.append("")
    if items is not None:
        lines.append("## Per-task audit")
        lines.extend(_task_table(analysis, items, scored))
        lines.append("")
    return "\n".join(lines) + "\n"


def build_analysis(
    prepared: dict[str, Any], rows: list[dict[str, Any]], gold_path: Path
) -> dict[str, Any]:
    items = list(prepared.get("items", []))
    scored = gold_path.is_file()
    items = attach_gold(items, gold_path)
    by_task = require_grid(items, rows)
    task_rows = [analyze_task(item, by_task[str(item["id"])]) for item in items]
    if not scored:
        for row in task_rows:
            row["exact_mode_correct"] = None
            row["selected_correct"] = None
            row["oracle_correct"] = False
            row["single_correct"] = False
    selectors = selector_summaries(task_rows)
    groups = source_groups(task_rows, items)
    by_source = {
        source: {
            "task_count": len(group),
            **selector_summaries(group),
        }
        for source, group in groups.items()
    }
    return {
        "schema_version": 1,
        "record_type": "external_transfer_analysis",
        "scored_against_gold": scored,
        "not_artifact_falsifier": True,
        "not_a_cybench_subtask_result": True,
        "support_threshold": SUPPORT_THRESHOLD,
        "task_count": len(task_rows),
        "source_counts": {source: len(group) for source, group in groups.items()},
        "tasks": task_rows,
        "selectors": selectors,
        "by_source": by_source,
        "bootstrap": {
            "selected_overall_accuracy": bootstrap_ci(
                task_rows, "selected_overall_accuracy"
            )
            if scored
            else None,
            "selective_risk": bootstrap_ci(task_rows, "selective_risk")
            if scored
            else None,
            "abstention_risk_reduction": bootstrap_ci(
                task_rows, "abstention_risk_reduction"
            )
            if scored
            else None,
        },
    }


def write_decisions(analysis: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "external_transfer_decisions",
        "scored_against_gold": False,
        "support_threshold": SUPPORT_THRESHOLD,
        "tasks": [
            {
                "task_id": row["task_id"],
                "exact_mode": row["exact_mode"],
                "exact_mode_support": row["exact_mode_support"],
                "abstained": row["abstained"],
                "selected": "" if row["abstained"] else row["exact_mode"],
                "valid_count": row["valid_count"],
                "invalid_count": row["invalid_count"],
            }
            for row in analysis["tasks"]
        ],
    }


def main() -> int:
    args = parse_args()
    prepared = load_json(Path(args.prepared))
    rows = load_jsonl(Path(args.raw))
    gold_path = Path(args.gold)
    if args.decisions_only and gold_path.is_file():
        gold_path = Path("/dev/null")
    if args.decisions_only:
        scored_path = Path("/nonexistent-external-gold")
        analysis = build_analysis(prepared, rows, scored_path)
    else:
        analysis = build_analysis(prepared, rows, gold_path)
    decisions = write_decisions(analysis)
    provenance = {
        "schema_version": 1,
        "prepared_sha256": sha256_file(Path(args.prepared)),
        "raw_sha256": sha256_file(Path(args.raw)),
        "freeze_sha256": sha256_file(Path("docs/external_method_freeze.md")),
        "scored_against_gold": analysis["scored_against_gold"],
    }
    summary = render_summary(
        analysis,
        analysis["scored_against_gold"],
        items=list(prepared.get("items", [])),
    )
    outputs = {
        Path(args.decisions): json.dumps(decisions, indent=2, sort_keys=True) + "\n",
        Path(args.output): json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        Path(args.summary): summary,
        Path(args.provenance): json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    }
    if args.validate_only:
        for path, expected in outputs.items():
            if path.exists() and path.read_text(encoding="utf-8") != expected:
                raise SystemExit(f"stale external transfer artifact: {path}")
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
