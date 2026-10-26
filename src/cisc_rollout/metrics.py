"""Task-level metrics and bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from random import Random
from statistics import mean
from typing import Any

from .normalize import normalize_answer
from .schema import Rollout, TaskItem
from .selectors import Selection, select_all


def _safe_div(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def evaluate_tasks(
    tasks: Sequence[TaskItem],
    rollouts_by_task: dict[str, Sequence[Rollout]],
    agreement_threshold: int = 3,
) -> dict[str, Any]:
    """Evaluate all fixed selectors at the parent-task level."""

    per_task: list[dict[str, Any]] = []
    selector_names = ["single", "majority", "tool_grounded_majority", "agreement_abstention", "oracle"]
    for task in tasks:
        rollouts = list(rollouts_by_task.get(task.id, ()))
        selections = select_all(
            rollouts, task.aliases, agreement_threshold=agreement_threshold
        )
        valid_answers = [normalize_answer(rollout.answer) for rollout in rollouts if rollout.valid and rollout.answer]
        per_task.append(
            {
                "task_id": task.id,
                "split": task.split,
                "selections": selections,
                "valid_rollouts": sum(1 for rollout in rollouts if rollout.valid),
                "parse_failures": sum(1 for rollout in rollouts if rollout.parse_failed),
                "tool_failures": sum(1 for rollout in rollouts if rollout.tool_failed),
                "unique_normalized_answers": len({answer for answer in valid_answers if answer}),
                "diversity_ratio": _safe_div(len({answer for answer in valid_answers if answer}), len(valid_answers)),
            }
        )
    summary: dict[str, Any] = {"n_tasks": len(per_task), "per_task": per_task, "selectors": {}}
    oracle_accuracy = _selector_overall(per_task, "oracle")
    for selector_name in selector_names:
        answered = sum(not row["selections"][selector_name].abstained for row in per_task)
        correct = sum(row["selections"][selector_name].correct is True for row in per_task)
        overall = _safe_div(correct, len(per_task))
        summary["selectors"][selector_name] = {
            "answered": answered,
            "correct": correct,
            "coverage": _safe_div(answered, len(per_task)),
            "selective_accuracy": _safe_div(correct, answered),
            "selective_risk": 1.0 - _safe_div(correct, answered) if answered else 0.0,
            "overall_accuracy": overall,
            "oracle_gap": oracle_accuracy - overall,
        }
    consensus_rows = [
        row
        for row in per_task
        if not row["selections"]["majority"].abstained
        and row["selections"]["majority"].support >= agreement_threshold
    ]
    wrong_consensus = sum(row["selections"]["majority"].correct is False for row in consensus_rows)
    total_rollouts = sum(len(rollouts_by_task.get(task.id, ())) for task in tasks)
    summary["secondary"] = {
        "wrong_consensus_rate": _safe_div(wrong_consensus, len(consensus_rows)),
        "mean_unique_normalized_answers": mean([row["unique_normalized_answers"] for row in per_task]) if per_task else 0.0,
        "mean_diversity_ratio": mean([row["diversity_ratio"] for row in per_task]) if per_task else 0.0,
        "parse_failure_rate": _safe_div(sum(row["parse_failures"] for row in per_task), total_rollouts),
        "tool_failure_rate": _safe_div(sum(row["tool_failures"] for row in per_task), total_rollouts),
    }
    return summary


def _selector_overall(per_task: Sequence[dict[str, Any]], selector_name: str) -> float:
    return _safe_div(sum(row["selections"][selector_name].correct is True for row in per_task), len(per_task))


def selector_stat(summary: dict[str, Any], selector_name: str, metric_name: str) -> float:
    return float(summary["selectors"][selector_name][metric_name])


def bootstrap_percentile_ci(
    tasks: Sequence[TaskItem],
    rollouts_by_task: dict[str, Sequence[Rollout]],
    statistic: Callable[[dict[str, Any]], float],
    samples: int = 2000,
    seed: int = 20260918,
    alpha: float = 0.05,
    agreement_threshold: int = 3,
) -> dict[str, float]:
    """Seeded nonparametric task bootstrap with percentile intervals."""

    if not tasks:
        raise ValueError("at least one task is required")
    rng = Random(seed)
    values: list[float] = []
    task_list = list(tasks)
    for _ in range(samples):
        resampled = [task_list[rng.randrange(len(task_list))] for _ in task_list]
        values.append(
            float(
                statistic(
                    evaluate_tasks(
                        resampled,
                        rollouts_by_task,
                        agreement_threshold=agreement_threshold,
                    )
                )
            )
        )
    values.sort()
    lower_index = int((alpha / 2.0) * (samples - 1))
    upper_index = int((1.0 - alpha / 2.0) * (samples - 1))
    return {
        "mean": mean(values),
        "lower": values[lower_index],
        "upper": values[upper_index],
        "samples": float(samples),
        "seed": float(seed),
    }


def paired_delta_stat(selector_name: str, metric_name: str = "overall_accuracy", baseline: str = "single") -> Callable[[dict[str, Any]], float]:
    def _stat(summary: dict[str, Any]) -> float:
        return selector_stat(summary, selector_name, metric_name) - selector_stat(summary, baseline, metric_name)

    return _stat
