#!/usr/bin/env python3
"""Analyze the frozen K=20 static CyBench holdout run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cisc_rollout.normalize import normalize_answer, normalized_aliases


SEEDS_20 = tuple(1101 * index for index in range(1, 21))
SUPPORT_THRESHOLD = 0.60
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260920


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default="data/holdout_prepared_items.json")
    parser.add_argument("--raw", default="data/raw/holdout_k20_rollouts.jsonl")
    parser.add_argument("--output", default="results/holdout_k20_metrics.json")
    parser.add_argument("--summary", default="results/holdout_k20_summary.md")
    parser.add_argument("--provenance", default="results/holdout_k20_provenance.json")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def final_text(record: dict[str, Any]) -> str:
    final = record.get("final")
    if isinstance(final, dict) and isinstance(final.get("final"), str):
        return final["final"]
    if isinstance(final, str):
        return final
    return ""


def normalized_final(record: dict[str, Any]) -> str:
    if bool(record.get("parse_failed")):
        return ""
    return normalize_answer(final_text(record))


def require_grid(items: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    item_ids = [str(item["id"]) for item in items]
    allowed = set(item_ids)
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id"))
        seed = int(row.get("seed", -1))
        key = (task_id, seed)
        if task_id not in allowed:
            raise ValueError(f"unexpected holdout task: {task_id}")
        if key in keyed:
            raise ValueError(f"duplicate holdout rollout: {task_id}:{seed}")
        keyed[key] = row
    missing = [
        f"{task_id}:{seed}"
        for task_id in item_ids
        for seed in SEEDS_20
        if (task_id, seed) not in keyed
    ]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... (+{len(missing) - 8})"
        raise ValueError(f"holdout raw grid incomplete: {preview}{suffix}")
    return {
        task_id: [keyed[(task_id, seed)] for seed in SEEDS_20]
        for task_id in item_ids
    }


def modal(counts: Counter[str]) -> tuple[str, int, float, bool]:
    valid_total = sum(counts.values())
    if valid_total == 0:
        return "", 0, 0.0, True
    top = max(counts.values())
    winners = sorted(answer for answer, count in counts.items() if count == top)
    return winners[0], top, top / valid_total, len(winners) > 1


def analyze_task(item: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aliases = normalized_aliases(tuple(item.get("gold_aliases", [])))
    counts = Counter(answer for row in rows if (answer := normalized_final(row)))
    mode, support_count, support, tied = modal(counts)
    invalid_count = len(rows) - sum(counts.values())
    mode_answered = bool(mode) and not tied
    mode_correct = None if not mode_answered else mode in aliases
    selected = ""
    abstained = True
    if mode and not tied and support >= SUPPORT_THRESHOLD:
        selected = mode
        abstained = False
    selected_correct = None if abstained else selected in aliases
    oracle_correct = any(answer in aliases for answer in counts)
    first_answer = normalized_final(rows[0])
    return {
        "task_id": item["id"],
        "source_task_path": item["source_task_path"],
        "valid_count": sum(counts.values()),
        "invalid_count": invalid_count,
        "invalid_rate": invalid_count / len(rows),
        "exact_mode": mode,
        "exact_mode_count": support_count,
        "exact_mode_support": support,
        "exact_mode_tied": tied,
        "exact_mode_correct": mode_correct,
        "abstained": abstained,
        "selected_correct": selected_correct,
        "oracle_correct": oracle_correct,
        "single_correct": bool(first_answer and first_answer in aliases),
        "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in rows),
        "wall_seconds": sum(float(row.get("wall_seconds") or 0.0) for row in rows),
    }


def selector_summaries(task_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    modal_answered = [
        row for row in task_rows if row["exact_mode_correct"] is not None
    ]
    modal_correct = sum(row["exact_mode_correct"] is True for row in modal_answered)
    modal_incorrect = sum(row["exact_mode_correct"] is False for row in modal_answered)
    answered = [row for row in task_rows if not row["abstained"]]
    correct = sum(row["selected_correct"] is True for row in answered)
    incorrect = sum(row["selected_correct"] is False for row in answered)
    single_correct = sum(row["single_correct"] for row in task_rows)
    oracle_correct = sum(row["oracle_correct"] for row in task_rows)
    reachable_headroom = oracle_correct - single_correct
    return {
        "single_seed_1101": {
            "correct": single_correct,
            "overall_accuracy": single_correct / len(task_rows),
        },
        "exact_modal_without_support_abstention": {
            "answered": len(modal_answered),
            "correct": modal_correct,
            "incorrect": modal_incorrect,
            "abstained_on_tie_or_no_valid": len(task_rows) - len(modal_answered),
            "overall_accuracy": modal_correct / len(task_rows),
            "risk": modal_incorrect / len(modal_answered) if modal_answered else None,
        },
        "exact_majority_support_ge_0_60": {
            "answered": len(answered),
            "correct": correct,
            "incorrect": incorrect,
            "abstained": len(task_rows) - len(answered),
            "coverage": len(answered) / len(task_rows),
            "selective_accuracy": correct / len(answered) if answered else 0.0,
            "selective_risk": incorrect / len(answered) if answered else None,
            "overall_accuracy": correct / len(task_rows),
        },
        "exact_oracle_pass_at_20": {
            "correct": oracle_correct,
            "overall_accuracy": oracle_correct / len(task_rows),
        },
        "reachable_headroom_recovery": {
            "single_correct": single_correct,
            "selected_correct": correct,
            "oracle_correct": oracle_correct,
            "reachable_headroom_tasks": reachable_headroom,
            "recovered_tasks": correct - single_correct,
            "recovered_fraction": (
                (correct - single_correct) / reachable_headroom
                if reachable_headroom > 0
                else None
            ),
        },
        "abstention_effect": {
            "wrong_modal_intercepted": sum(
                row["exact_mode_correct"] is False and row["abstained"]
                for row in task_rows
            ),
            "correct_modal_abstained": sum(
                row["exact_mode_correct"] is True and row["abstained"]
                for row in task_rows
            ),
            "risk_reduction": (
                modal_incorrect / len(modal_answered) - incorrect / len(answered)
                if modal_answered and answered
                else None
            ),
        },
    }


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1))
    return ordered[index]


def bootstrap_ci(
    task_rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    rng = Random(seed)
    values: list[float] = []
    rows = list(task_rows)
    for _ in range(samples):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        selectors = selector_summaries(sample)
        if metric == "selected_overall_accuracy":
            value = selectors["exact_majority_support_ge_0_60"]["overall_accuracy"]
        elif metric == "exact_modal_accuracy":
            value = selectors["exact_modal_without_support_abstention"][
                "overall_accuracy"
            ]
        elif metric == "oracle_accuracy":
            value = selectors["exact_oracle_pass_at_20"]["overall_accuracy"]
        elif metric == "selective_risk":
            risk = selectors["exact_majority_support_ge_0_60"]["selective_risk"]
            if risk is None:
                continue
            value = risk
        elif metric == "abstention_risk_reduction":
            reduction = selectors["abstention_effect"]["risk_reduction"]
            if reduction is None:
                continue
            value = reduction
        else:
            raise ValueError(f"unsupported bootstrap metric: {metric}")
        values.append(float(value))
    return {
        "lower": percentile(values, 0.025) if values else None,
        "upper": percentile(values, 0.975) if values else None,
        "samples": samples,
        "defined_samples": len(values),
        "seed": seed,
    }


def build_analysis(prepared: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = list(prepared.get("items", []))
    if not items:
        raise ValueError("holdout prepared file contains no items")
    by_task = require_grid(items, rows)
    task_rows = [analyze_task(item, by_task[str(item["id"])]) for item in items]
    selectors = selector_summaries(task_rows)
    return {
        "schema_version": 1,
        "analysis": "holdout_k20_frozen_exact_majority",
        "task_count": len(items),
        "rollout_count": len(rows),
        "support_threshold": SUPPORT_THRESHOLD,
        "selectors": selectors,
        "bootstrap_95_ci": {
            "selected_overall_accuracy": bootstrap_ci(
                task_rows, "selected_overall_accuracy"
            ),
            "exact_modal_accuracy": bootstrap_ci(task_rows, "exact_modal_accuracy"),
            "oracle_accuracy": bootstrap_ci(task_rows, "oracle_accuracy"),
            "selective_risk": bootstrap_ci(task_rows, "selective_risk"),
            "abstention_risk_reduction": bootstrap_ci(
                task_rows, "abstention_risk_reduction"
            ),
        },
        "invalid_rate": sum(row["invalid_count"] for row in task_rows) / len(rows),
        "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in rows),
        "mean_generated_tokens_per_rollout": sum(
            int(row.get("generated_tokens") or 0) for row in rows
        )
        / len(rows),
        "wall_seconds": sum(float(row.get("wall_seconds") or 0.0) for row in rows),
        "mean_wall_seconds_per_rollout": sum(
            float(row.get("wall_seconds") or 0.0) for row in rows
        )
        / len(rows),
        "task_rows": task_rows,
    }


def build_summary(analysis: dict[str, Any]) -> str:
    majority = analysis["selectors"]["exact_majority_support_ge_0_60"]
    modal = analysis["selectors"]["exact_modal_without_support_abstention"]
    oracle = analysis["selectors"]["exact_oracle_pass_at_20"]
    single = analysis["selectors"]["single_seed_1101"]
    headroom = analysis["selectors"]["reachable_headroom_recovery"]
    abstention = analysis["selectors"]["abstention_effect"]
    lines = [
        "# Frozen holdout K=20 summary",
        "",
        f"Tasks: {analysis['task_count']}; rollouts: {analysis['rollout_count']}.",
        f"Single-seed exact accuracy: {single['correct']}/{analysis['task_count']}.",
        f"Exact modal accuracy without support abstention: {modal['correct']}/{analysis['task_count']} ({modal['answered']} uniquely answered).",
        f"Exact majority at support >= 0.60: answered {majority['answered']}, correct {majority['correct']}, incorrect {majority['incorrect']}, abstained {majority['abstained']}.",
        f"Abstention intercepted {abstention['wrong_modal_intercepted']} wrong modals and abstained on {abstention['correct_modal_abstained']} correct modals; descriptive risk reduction was {abstention['risk_reduction']:.3f}.",
        f"Exact Oracle Pass@20: {oracle['correct']}/{analysis['task_count']}.",
        f"Recovered reachable headroom: {headroom['recovered_tasks']}/{headroom['reachable_headroom_tasks']} ({headroom['recovered_fraction']:.3f}).",
        f"Invalid rollout rate: {analysis['invalid_rate']:.3f}.",
        f"Mean generated tokens per rollout: {analysis['mean_generated_tokens_per_rollout']:.1f}.",
        f"Mean wall seconds per rollout: {analysis['mean_wall_seconds_per_rollout']:.2f}.",
        f"Selected overall accuracy bootstrap 95% CI: [{analysis['bootstrap_95_ci']['selected_overall_accuracy']['lower']:.3f}, {analysis['bootstrap_95_ci']['selected_overall_accuracy']['upper']:.3f}].",
        f"Exact modal accuracy bootstrap 95% CI: [{analysis['bootstrap_95_ci']['exact_modal_accuracy']['lower']:.3f}, {analysis['bootstrap_95_ci']['exact_modal_accuracy']['upper']:.3f}].",
        f"Oracle accuracy bootstrap 95% CI: [{analysis['bootstrap_95_ci']['oracle_accuracy']['lower']:.3f}, {analysis['bootstrap_95_ci']['oracle_accuracy']['upper']:.3f}].",
        f"Abstention risk-reduction bootstrap 95% CI: [{analysis['bootstrap_95_ci']['abstention_risk_reduction']['lower']:.3f}, {analysis['bootstrap_95_ci']['abstention_risk_reduction']['upper']:.3f}].",
        "",
        "| Task | Modal | Support | Modal correct | Selected | Oracle | Invalid |",
        "| --- | --- | ---: | --- | --- | --- | ---: |",
    ]
    for row in analysis["task_rows"]:
        selected = "abstain" if row["abstained"] else str(row["selected_correct"])
        lines.append(
            f"| {row['task_id']} | `{row['exact_mode']}` | "
            f"{row['exact_mode_support']:.2f} | {row['exact_mode_correct']} | {selected} | "
            f"{row['oracle_correct']} | {row['invalid_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def build_provenance(prepared_path: Path, raw_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "holdout_k20_analysis",
        "prepared_sha256": sha256_file(prepared_path),
        "raw_sha256": sha256_file(raw_path),
        "method_freeze_sha256": sha256_file(Path("docs/method_freeze.md")),
        "seeds": list(SEEDS_20),
        "support_threshold": SUPPORT_THRESHOLD,
    }


def main() -> int:
    args = parse_args()
    prepared_path = Path(args.prepared)
    raw_path = Path(args.raw)
    output_path = Path(args.output)
    summary_path = Path(args.summary)
    provenance_path = Path(args.provenance)
    prepared = load_json(prepared_path)
    rows = load_jsonl(raw_path)
    analysis = build_analysis(prepared, rows)
    summary = build_summary(analysis)
    provenance = build_provenance(prepared_path, raw_path)
    if args.validate_only:
        if output_path.exists() and load_json(output_path) != analysis:
            raise SystemExit(f"stale holdout metrics: {output_path}")
        if summary_path.exists() and summary_path.read_text(encoding="utf-8") != summary:
            raise SystemExit(f"stale holdout summary: {summary_path}")
        if provenance_path.exists() and load_json(provenance_path) != provenance:
            raise SystemExit(f"stale holdout provenance: {provenance_path}")
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(analysis["selectors"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
