#!/usr/bin/env python3
"""Create and validate the exact K=25 prefix of the real K=50 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


SEEDS = tuple(1101 * index for index in range(1, 26))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw/main_k50_rollouts.jsonl")
    parser.add_argument("--baseline", default="data/raw/main_rollouts.jsonl")
    parser.add_argument("--output", default="data/raw/main_k25_rollouts.jsonl")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def read_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        rows.append((line, record))
    return rows


def selected_source_lines(source: Path) -> list[tuple[str, dict[str, Any]]]:
    selected = [
        (line, record)
        for line, record in read_lines(source)
        if record.get("split") == "main" and int(record.get("seed", -1)) in SEEDS
    ]
    pairs = [(str(record.get("task_id")), int(record.get("seed", -1))) for _, record in selected]
    task_ids = {task_id for task_id, _ in pairs}
    errors: list[str] = []
    if len(selected) != 375:
        errors.append(f"expected 375 selected records, found {len(selected)}")
    if len(task_ids) != 15:
        errors.append(f"expected 15 tasks, found {len(task_ids)}")
    if len(set(pairs)) != len(pairs):
        errors.append("duplicate task/seed pair in selected source")
    expected = {(task_id, seed) for task_id in task_ids for seed in SEEDS}
    if set(pairs) != expected:
        errors.append("selected source does not contain the complete 15x25 grid")
    if any(record.get("evidence_kind") != "real_model_benchmark" for _, record in selected):
        errors.append("selected source contains non-real evidence")
    if any(bool(record.get("infrastructure_failed")) for _, record in selected):
        errors.append("selected source contains an infrastructure failure")
    if errors:
        raise ValueError("; ".join(errors))
    return selected


def validate_baseline(
    selected: list[tuple[str, dict[str, Any]]], baseline_path: Path
) -> None:
    selected_by_pair = {
        (str(record["task_id"]), int(record["seed"])): record
        for _, record in selected
    }
    baseline = [record for _, record in read_lines(baseline_path)]
    if len(baseline) != 75:
        raise ValueError(f"expected 75 K=5 baseline records, found {len(baseline)}")
    for record in baseline:
        pair = (str(record["task_id"]), int(record["seed"]))
        if selected_by_pair.get(pair) != record:
            raise ValueError(f"K=5 baseline record changed: {pair}")


def main() -> int:
    args = parse_args()
    source = Path(args.source)
    baseline = Path(args.baseline)
    output = Path(args.output)
    selected = selected_source_lines(source)
    validate_baseline(selected, baseline)
    expected_text = "\n".join(line for line, _ in selected) + "\n"
    if args.validate_only:
        if not output.exists():
            raise SystemExit(f"K=25 raw evidence is missing: {output}")
        if output.read_text(encoding="utf-8") != expected_text:
            raise SystemExit("K=25 raw evidence is not the exact K=50 prefix subset")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks": 15,
                "rollouts_per_task": 25,
                "rollout_count": len(selected),
                "source": str(source),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
