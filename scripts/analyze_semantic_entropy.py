#!/usr/bin/env python3
"""Measure exact and empirical semantic entropy on the frozen K=50 rollouts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
from random import Random
import re
import sys
from typing import Any

from cisc_rollout.normalize import normalize_answer
SEEDS = tuple(1101 * index for index in range(1, 51))
CHECKPOINTS = (5, 10, 15, 25, 50)
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20260920
CORRECTNESS = {"correct", "incorrect", "ambiguous"}


def shannon_entropy(counts: Iterable[int]) -> float:
    """Return empirical Shannon entropy in nats for positive category counts."""

    positive = [int(count) for count in counts if int(count) > 0]
    total = sum(positive)
    if total <= 1 or len(positive) <= 1:
        return 0.0
    return -sum((count / total) * math.log(count / total) for count in positive)


def normalized_entropy(counts: Iterable[int]) -> float:
    """Normalize empirical entropy by the maximum log(N) for N observations."""

    positive = [int(count) for count in counts if int(count) > 0]
    total = sum(positive)
    if total <= 1:
        return 0.0
    return shannon_entropy(positive) / math.log(total)


def binary_auroc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    """Compute AUROC by pairwise ranking, assigning half credit to score ties."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def grouped_risk_coverage(
    labels: Sequence[bool], uncertainties: Sequence[float]
) -> tuple[list[dict[str, float | int]], float]:
    """Accept low-uncertainty tie groups together and return points plus AURC."""

    if len(labels) != len(uncertainties):
        raise ValueError("labels and uncertainties must have equal length")
    if not labels:
        return [], 0.0
    thresholds = sorted(set(float(value) for value in uncertainties))
    points: list[dict[str, float | int]] = []
    previous_coverage = 0.0
    aurc = 0.0
    for threshold in thresholds:
        accepted = [
            label
            for label, uncertainty in zip(labels, uncertainties)
            if float(uncertainty) <= threshold
        ]
        coverage = len(accepted) / len(labels)
        risk = sum(not label for label in accepted) / len(accepted)
        aurc += risk * (coverage - previous_coverage)
        previous_coverage = coverage
        points.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "coverage": coverage,
                "risk": risk,
            }
        )
    return points, aurc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/main_k50_rollouts.jsonl")
    parser.add_argument("--manifest", default="data/task_manifest.json")
    parser.add_argument(
        "--annotations", default="data/annotations/k50_semantic_clusters.json"
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        records.append(value)
    return records


def final_answer(record: dict[str, Any]) -> str | None:
    final = record.get("final")
    if isinstance(final, dict) and isinstance(final.get("final"), str):
        return final["final"]
    if isinstance(final, str):
        return final
    return None


def validate_grid(
    records: list[dict[str, Any]], task_ids: list[str]
) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {(task_id, seed) for task_id in task_ids for seed in SEEDS}
    keyed: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    for record in records:
        pair = (str(record.get("task_id")), int(record.get("seed", -1)))
        if pair in keyed:
            errors.append(f"duplicate task/seed pair: {pair}")
        keyed[pair] = record
        if record.get("evidence_kind") != "real_model_benchmark":
            errors.append(f"non-real evidence: {pair}")
        if record.get("model") != "granite-4.2-8b-Q4_K_M":
            errors.append(f"unexpected model: {pair}")
        if record.get("temperature") != 1.0 or record.get("top_p") != 0.95:
            errors.append(f"unexpected sampling settings: {pair}")
        if bool(record.get("infrastructure_failed")):
            errors.append(f"infrastructure failure: {pair}")
    if set(keyed) != expected:
        errors.append(
            f"K=50 grid mismatch: missing={len(expected - set(keyed))}, "
            f"extra={len(set(keyed) - expected)}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return keyed


def validate_annotations(annotations: dict[str, Any], task_ids: list[str]) -> None:
    task_rules = annotations.get("tasks")
    if not isinstance(task_rules, dict) or set(task_rules) != set(task_ids):
        present = set(task_rules) if isinstance(task_rules, dict) else set()
        raise ValueError(
            f"annotation task mismatch: missing={set(task_ids) - present}, "
            f"extra={present - set(task_ids)}"
        )
    for task_id, task_rule in task_rules.items():
        clusters = task_rule.get("clusters") if isinstance(task_rule, dict) else None
        if not isinstance(clusters, list) or not clusters:
            raise ValueError(f"missing semantic clusters: {task_id}")
        ids: set[str] = set()
        for cluster in clusters:
            cluster_id = cluster.get("id")
            if not isinstance(cluster_id, str) or not cluster_id or cluster_id in ids:
                raise ValueError(f"invalid or duplicate cluster id for {task_id}")
            ids.add(cluster_id)
            if cluster.get("correctness") not in CORRECTNESS:
                raise ValueError(f"invalid correctness for {task_id}/{cluster_id}")
            patterns = cluster.get("patterns")
            if not isinstance(patterns, list) or not patterns:
                raise ValueError(f"missing patterns for {task_id}/{cluster_id}")
            for pattern in patterns:
                if not isinstance(pattern, str):
                    raise ValueError(f"non-string pattern for {task_id}/{cluster_id}")
                re.compile(pattern)


def classify_semantic_answer(
    task_id: str, normalized_answer: str, annotations: dict[str, Any]
) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    clusters = annotations["tasks"][task_id]["clusters"]
    for cluster in clusters:
        if any(re.fullmatch(pattern, normalized_answer) for pattern in cluster["patterns"]):
            matches.append(cluster)
    if len(matches) != 1:
        ids = [cluster["id"] for cluster in matches]
        raise ValueError(
            f"semantic answer must match exactly one cluster: task={task_id!r}, "
            f"answer={normalized_answer!r}, matches={ids}"
        )
    match = matches[0]
    return {
        "cluster_id": match["id"],
        "cluster_label": match["label"],
        "correctness": match["correctness"],
    }


def modal_summary(
    counts: Counter[str], correctness_by_value: dict[str, str]
) -> dict[str, Any]:
    if not counts:
        return {
            "ids": [],
            "count": 0,
            "support": 0.0,
            "correctness": "ambiguous",
        }
    modal_count = max(counts.values())
    modal_ids = sorted(value for value, count in counts.items() if count == modal_count)
    correctness = {correctness_by_value[value] for value in modal_ids}
    return {
        "ids": modal_ids,
        "count": modal_count,
        "support": modal_count / sum(counts.values()),
        "correctness": correctness.pop() if len(correctness) == 1 else "ambiguous",
    }


def analyze_prefix(
    task_id: str,
    records: list[dict[str, Any]],
    annotations: dict[str, Any],
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    semantic_correctness: dict[str, str] = {}
    exact_correctness: dict[str, str] = {}
    assignment_rows: list[dict[str, Any]] = []

    for record in records[:k]:
        answer = final_answer(record)
        normalized = normalize_answer(answer) if answer is not None else ""
        valid = bool(normalized) and not bool(record.get("parse_failed"))
        if not valid:
            continue
        semantic = classify_semantic_answer(task_id, normalized, annotations)
        exact_counts[normalized] += 1
        semantic_counts[semantic["cluster_id"]] += 1
        semantic_correctness[semantic["cluster_id"]] = semantic["correctness"]
        exact_correctness[normalized] = semantic["correctness"]
        assignment_rows.append(
            {
                "seed": int(record["seed"]),
                "normalized_answer": normalized,
                **semantic,
            }
        )

    valid_count = sum(exact_counts.values())
    no_answer_count = k - valid_count
    exact_inclusive = list(exact_counts.values()) + ([no_answer_count] if no_answer_count else [])
    semantic_inclusive = list(semantic_counts.values()) + (
        [no_answer_count] if no_answer_count else []
    )
    semantic_modal = modal_summary(semantic_counts, semantic_correctness)
    exact_modal = modal_summary(exact_counts, exact_correctness)
    correct_cluster_present = any(
        semantic_correctness[cluster_id] == "correct" for cluster_id in semantic_counts
    )
    return (
        {
            "task_id": task_id,
            "k": k,
            "valid_count": valid_count,
            "no_answer_count": no_answer_count,
            "exact_cluster_count": len(exact_counts),
            "semantic_cluster_count": len(semantic_counts),
            "exact_entropy_nats_valid": shannon_entropy(exact_counts.values()),
            "semantic_entropy_nats_valid": shannon_entropy(semantic_counts.values()),
            "exact_entropy_normalized_valid": normalized_entropy(exact_counts.values()),
            "semantic_entropy_normalized_valid": normalized_entropy(semantic_counts.values()),
            "exact_entropy_nats_inclusive": shannon_entropy(exact_inclusive),
            "semantic_entropy_nats_inclusive": shannon_entropy(semantic_inclusive),
            "exact_entropy_normalized_inclusive": normalized_entropy(exact_inclusive),
            "semantic_entropy_normalized_inclusive": normalized_entropy(semantic_inclusive),
            "exact_modal_ids": exact_modal["ids"],
            "exact_modal_support_valid": exact_modal["support"],
            "exact_modal_correctness": exact_modal["correctness"],
            "semantic_modal_cluster_ids": semantic_modal["ids"],
            "semantic_modal_support_valid": semantic_modal["support"],
            "semantic_variation_ratio_valid": 1.0 - semantic_modal["support"],
            "semantic_modal_correctness": semantic_modal["correctness"],
            "semantic_correct_cluster_present": correct_cluster_present,
            "semantic_unanimous_valid": valid_count > 0 and len(semantic_counts) == 1,
        },
        assignment_rows,
    )


def percentile_interval(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "bootstrap_mean": sum(ordered) / len(ordered),
        "lower": ordered[int(0.025 * (len(ordered) - 1))],
        "upper": ordered[int(0.975 * (len(ordered) - 1))],
    }


def stratified_bootstrap_auroc(
    rows: list[dict[str, Any]],
    score_key: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    correct = [row for row in rows if row["semantic_modal_correctness"] == "correct"]
    incorrect = [row for row in rows if row["semantic_modal_correctness"] == "incorrect"]
    if not correct or not incorrect:
        raise ValueError("bootstrap AUROC requires correct and incorrect modal tasks")
    rng = Random(seed)
    estimates: list[float] = []
    for _ in range(samples):
        sampled_correct = [correct[rng.randrange(len(correct))] for _ in correct]
        sampled_incorrect = [incorrect[rng.randrange(len(incorrect))] for _ in incorrect]
        sampled = sampled_correct + sampled_incorrect
        estimates.append(
            binary_auroc(
                [row["semantic_modal_correctness"] == "correct" for row in sampled],
                [float(row[score_key]) for row in sampled],
            )
        )
    return percentile_interval(estimates)


def evaluate_signal(
    rows: list[dict[str, Any]],
    name: str,
    score_key: str,
    uncertainty_key: str,
    bootstrap_samples: int,
    seed_offset: int,
) -> dict[str, Any]:
    evaluable = [
        row
        for row in rows
        if row["semantic_modal_correctness"] in {"correct", "incorrect"}
    ]
    labels = [row["semantic_modal_correctness"] == "correct" for row in evaluable]
    scores = [float(row[score_key]) for row in evaluable]
    uncertainties = [float(row[uncertainty_key]) for row in evaluable]
    risk_coverage, aurc = grouped_risk_coverage(labels, uncertainties)
    return {
        "name": name,
        "evaluable_tasks": len(evaluable),
        "correct_tasks": sum(labels),
        "incorrect_tasks": sum(not label for label in labels),
        "auroc": binary_auroc(labels, scores),
        "auroc_stratified_task_bootstrap": stratified_bootstrap_auroc(
            evaluable,
            score_key,
            bootstrap_samples,
            BOOTSTRAP_SEED + seed_offset,
        ),
        "aurc_grouped_ties": aurc,
        "risk_coverage": risk_coverage,
    }


def mean_value(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def build_analysis(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    annotations: dict[str, Any],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_items = [item for item in manifest["items"] if item["split"] == "main"]
    task_ids = [str(item["id"]) for item in task_items]
    validate_annotations(annotations, task_ids)
    keyed = validate_grid(records, task_ids)

    all_rows: list[dict[str, Any]] = []
    final_assignments: list[dict[str, Any]] = []
    checkpoint_summary: list[dict[str, Any]] = []
    for k in CHECKPOINTS:
        checkpoint_rows: list[dict[str, Any]] = []
        for task_id in task_ids:
            ordered = [keyed[(task_id, seed)] for seed in SEEDS]
            row, assignments = analyze_prefix(task_id, ordered, annotations, k)
            checkpoint_rows.append(row)
            all_rows.append(row)
            if k == 50:
                counts: Counter[tuple[str, str, str, str]] = Counter(
                    (
                        assignment["normalized_answer"],
                        assignment["cluster_id"],
                        assignment["cluster_label"],
                        assignment["correctness"],
                    )
                    for assignment in assignments
                )
                final_assignments.append(
                    {
                        "task_id": task_id,
                        "assignments": [
                            {
                                "normalized_answer": answer,
                                "count": count,
                                "cluster_id": cluster_id,
                                "cluster_label": cluster_label,
                                "correctness": correctness,
                            }
                            for (
                                answer,
                                cluster_id,
                                cluster_label,
                                correctness,
                            ), count in sorted(counts.items())
                        ],
                    }
                )
        checkpoint_summary.append(
            {
                "k": k,
                "semantic_oracle_tasks": sum(
                    bool(row["semantic_correct_cluster_present"])
                    for row in checkpoint_rows
                ),
                "semantic_modal_correct": sum(
                    row["semantic_modal_correctness"] == "correct"
                    for row in checkpoint_rows
                ),
                "semantic_modal_incorrect": sum(
                    row["semantic_modal_correctness"] == "incorrect"
                    for row in checkpoint_rows
                ),
                "semantic_modal_ambiguous": sum(
                    row["semantic_modal_correctness"] == "ambiguous"
                    for row in checkpoint_rows
                ),
                "mean_exact_entropy_nats_valid": mean_value(
                    checkpoint_rows, "exact_entropy_nats_valid"
                ),
                "mean_semantic_entropy_nats_valid": mean_value(
                    checkpoint_rows, "semantic_entropy_nats_valid"
                ),
            }
        )

    final_rows = [row for row in all_rows if row["k"] == 50]
    for row in final_rows:
        row["negative_exact_entropy_score"] = -float(row["exact_entropy_nats_valid"])
        row["negative_semantic_entropy_score"] = -float(
            row["semantic_entropy_nats_valid"]
        )
        row["negative_semantic_entropy_inclusive_score"] = -float(
            row["semantic_entropy_nats_inclusive"]
        )
        row["semantic_modal_support_score"] = float(
            row["semantic_modal_support_valid"]
        )
        row["semantic_support_uncertainty"] = 1.0 - float(
            row["semantic_modal_support_valid"]
        )

    signals = {
        "exact_entropy": evaluate_signal(
            final_rows,
            "negative exact-answer entropy",
            "negative_exact_entropy_score",
            "exact_entropy_nats_valid",
            bootstrap_samples,
            0,
        ),
        "semantic_entropy": evaluate_signal(
            final_rows,
            "negative empirical semantic entropy",
            "negative_semantic_entropy_score",
            "semantic_entropy_nats_valid",
            bootstrap_samples,
            1,
        ),
        "semantic_entropy_inclusive": evaluate_signal(
            final_rows,
            "negative empirical semantic entropy including no-answer outcomes",
            "negative_semantic_entropy_inclusive_score",
            "semantic_entropy_nats_inclusive",
            bootstrap_samples,
            2,
        ),
        "semantic_modal_support": evaluate_signal(
            final_rows,
            "semantic modal support",
            "semantic_modal_support_score",
            "semantic_support_uncertainty",
            bootstrap_samples,
            3,
        ),
    }
    incorrect_rows = [
        row for row in final_rows if row["semantic_modal_correctness"] == "incorrect"
    ]
    correct_minority = [
        row["task_id"]
        for row in incorrect_rows
        if bool(row["semantic_correct_cluster_present"])
    ]
    wrong_unanimous = [
        row["task_id"]
        for row in incorrect_rows
        if bool(row["semantic_unanimous_valid"])
    ]
    wrong_ge_08 = [
        row["task_id"]
        for row in incorrect_rows
        if float(row["semantic_modal_support_valid"]) >= 0.8
    ]
    wrong_ge_09 = [
        row["task_id"]
        for row in incorrect_rows
        if float(row["semantic_modal_support_valid"]) >= 0.9
    ]
    metrics = {
        "schema_version": 1,
        "evidence_kind": "real_model_benchmark_post_hoc_analysis",
        "analysis": "k50_empirical_semantic_entropy",
        "analysis_date": "2026-09-20",
        "task_count": len(task_ids),
        "rollout_count": len(records),
        "rollouts_per_task": 50,
        "sampling": {"temperature": 1.0, "top_p": 0.95},
        "estimator": "frequency_based_empirical_entropy_nats",
        "annotation_status": annotations["annotation_status"],
        "checkpoints": checkpoint_summary,
        "primary_k50": {
            "modal_correct": sum(
                row["semantic_modal_correctness"] == "correct" for row in final_rows
            ),
            "modal_incorrect": len(incorrect_rows),
            "modal_ambiguous": sum(
                row["semantic_modal_correctness"] == "ambiguous" for row in final_rows
            ),
            "semantic_oracle_tasks": sum(
                bool(row["semantic_correct_cluster_present"]) for row in final_rows
            ),
            "tasks_with_exact_to_semantic_cluster_collapse": sum(
                int(row["exact_cluster_count"]) > int(row["semantic_cluster_count"])
                for row in final_rows
            ),
            "mean_exact_entropy_nats_valid": mean_value(
                final_rows, "exact_entropy_nats_valid"
            ),
            "mean_semantic_entropy_nats_valid": mean_value(
                final_rows, "semantic_entropy_nats_valid"
            ),
            "mean_entropy_reduction_nats": mean_value(
                final_rows, "exact_entropy_nats_valid"
            )
            - mean_value(final_rows, "semantic_entropy_nats_valid"),
            "wrong_semantic_unanimity": {
                "count": len(wrong_unanimous),
                "task_ids": wrong_unanimous,
            },
            "wrong_modal_support_ge_0_8": {
                "count": len(wrong_ge_08),
                "task_ids": wrong_ge_08,
            },
            "wrong_modal_support_ge_0_9": {
                "count": len(wrong_ge_09),
                "task_ids": wrong_ge_09,
            },
            "wrong_modal_with_correct_minority": {
                "count": len(correct_minority),
                "task_ids": correct_minority,
            },
            "signals": signals,
        },
        "task_rows": final_rows,
    }
    assignments = {
        "schema_version": 1,
        "analysis": "k50_empirical_semantic_entropy",
        "normalization": annotations["normalization"],
        "annotation_status": annotations["annotation_status"],
        "tasks": final_assignments,
    }
    return metrics, assignments


def render_csv(metrics: dict[str, Any]) -> str:
    fields = [
        "task_id",
        "valid_count",
        "no_answer_count",
        "exact_cluster_count",
        "semantic_cluster_count",
        "exact_entropy_nats_valid",
        "semantic_entropy_nats_valid",
        "exact_entropy_nats_inclusive",
        "semantic_entropy_nats_inclusive",
        "semantic_modal_cluster_ids",
        "semantic_modal_support_valid",
        "semantic_variation_ratio_valid",
        "semantic_modal_correctness",
        "semantic_correct_cluster_present",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(
        handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    for row in metrics["task_rows"]:
        serializable = dict(row)
        serializable["semantic_modal_cluster_ids"] = "|".join(
            row["semantic_modal_cluster_ids"]
        )
        writer.writerow(serializable)
    return handle.getvalue()


def render_summary(metrics: dict[str, Any]) -> str:
    primary = metrics["primary_k50"]
    exact = primary["signals"]["exact_entropy"]
    semantic = primary["signals"]["semantic_entropy"]
    support = primary["signals"]["semantic_modal_support"]
    semantic_inclusive = primary["signals"]["semantic_entropy_inclusive"]
    rows = metrics["task_rows"]
    checkpoints = [
        "| K | Semantic oracle | Modal correct | Modal incorrect | Modal ambiguous | Mean H semantic |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for checkpoint in metrics["checkpoints"]:
        checkpoints.append(
            "| {k} | {oracle}/15 | {correct}/15 | {incorrect}/15 | {ambiguous}/15 | {entropy:.3f} |".format(
                k=checkpoint["k"],
                oracle=checkpoint["semantic_oracle_tasks"],
                correct=checkpoint["semantic_modal_correct"],
                incorrect=checkpoint["semantic_modal_incorrect"],
                ambiguous=checkpoint["semantic_modal_ambiguous"],
                entropy=checkpoint["mean_semantic_entropy_nats_valid"],
            )
        )
    table = [
        "| Task | H exact | H semantic | Modal support | Modal label | Correct candidate present |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        table.append(
            "| {task} | {exact:.3f} | {semantic:.3f} | {support:.3f} | {label} | {oracle} |".format(
                task=row["task_id"].removeprefix("main-"),
                exact=row["exact_entropy_nats_valid"],
                semantic=row["semantic_entropy_nats_valid"],
                support=row["semantic_modal_support_valid"],
                label=row["semantic_modal_correctness"],
                oracle="yes" if row["semantic_correct_cluster_present"] else "no",
            )
        )
    wrong_unanimous = primary["wrong_semantic_unanimity"]
    wrong_09 = primary["wrong_modal_support_ge_0_9"]
    minority = primary["wrong_modal_with_correct_minority"]
    semantic_first = semantic["risk_coverage"][0]
    return f"""# K=50 empirical semantic-entropy results

## Result

This retrospective analysis used all 750 existing Granite rollouts and no new
model inference.  Manual hybrid semantic clustering reduced mean entropy from
{primary['mean_exact_entropy_nats_valid']:.3f} nats for normalized strings to
{primary['mean_semantic_entropy_nats_valid']:.3f} nats for semantic clusters;
{primary['tasks_with_exact_to_semantic_cluster_collapse']}/15 tasks collapsed at
least two exact strings into fewer meanings.

At K=50, the semantic modal answer was annotated correct on
{primary['modal_correct']}/15 tasks, incorrect on {primary['modal_incorrect']}/15,
and ambiguous on {primary['modal_ambiguous']}/15.  A correct semantic candidate
appeared on {primary['semantic_oracle_tasks']}/15 tasks.  The latter is
reachability, not deployable accuracy.

## Does entropy identify correctness?

Using lower uncertainty as greater confidence and excluding the one ambiguous
modal task:

| Signal | AUROC | 95% stratified task-bootstrap interval | Grouped-tie AURC | 
|---|---:|---:|---:|
| Exact-answer entropy | {exact['auroc']:.3f} | [{exact['auroc_stratified_task_bootstrap']['lower']:.3f}, {exact['auroc_stratified_task_bootstrap']['upper']:.3f}] | {exact['aurc_grouped_ties']:.3f} |
| Empirical semantic entropy | {semantic['auroc']:.3f} | [{semantic['auroc_stratified_task_bootstrap']['lower']:.3f}, {semantic['auroc_stratified_task_bootstrap']['upper']:.3f}] | {semantic['aurc_grouped_ties']:.3f} |
| Semantic entropy including no-answer | {semantic_inclusive['auroc']:.3f} | [{semantic_inclusive['auroc_stratified_task_bootstrap']['lower']:.3f}, {semantic_inclusive['auroc_stratified_task_bootstrap']['upper']:.3f}] | {semantic_inclusive['aurc_grouped_ties']:.3f} |
| Semantic modal support | {support['auroc']:.3f} | [{support['auroc_stratified_task_bootstrap']['lower']:.3f}, {support['auroc_stratified_task_bootstrap']['upper']:.3f}] | {support['aurc_grouped_ties']:.3f} |

The lowest semantic-entropy acceptance group covered
{semantic_first['coverage']:.1%} of evaluable tasks at
{semantic_first['risk']:.1%} selective risk.  Thus zero empirical semantic
entropy did not imply zero error.

## Prefix behavior

{chr(10).join(checkpoints)}

The semantic oracle increased as additional samples exposed correct minority
candidates, but modal correctness did not improve monotonically.  The K=10 row
uses the same number of samples as the cited Semantic Entropy baseline, but it
is not a reproduction because the sampling temperature, top-p, tasks, and
semantic classifier differ.

## Low-entropy wrong consensus

- Semantic unanimity among valid rollouts with a wrong modal cluster occurred
  on {wrong_unanimous['count']} task(s): {', '.join(wrong_unanimous['task_ids']) or 'none'}.
- Wrong modal support of at least 0.9 occurred on {wrong_09['count']} task(s):
  {', '.join(wrong_09['task_ids']) or 'none'}.
- A correct minority cluster existed despite an incorrect modal cluster on
  {minority['count']} task(s): {', '.join(minority['task_ids']) or 'none'}.

The MOTP result is the clearest near-unanimous failure: 49/50 valid answers are
`!==`, while the single `==` answer is correct.  Its empirical semantic entropy
is low, so an entropy-only abstention policy treats this failure as confident.

## Per-task audit table

{chr(10).join(table)}

## Interpretation limits

This is a frequency-based black-box estimator, not a likelihood-weighted
reproduction of the original Semantic Entropy implementation.  The original
paper's formal self-consistent-error definition is also not applied because
there is no separate greedy T=0 response.  Semantic clusters were created
post-hoc by one annotator with access to reference answers, and the correctness
AUROC uses only 14 evaluable parent tasks.  These results support a pilot claim
and experimental design, not a calibrated deployment threshold.  Independent
cluster adjudication and a larger held-out task set are required for a paper's
confirmatory result.
"""


def render_svg(metrics: dict[str, Any]) -> str:
    width, height = 760, 500
    left, top, plot_width, plot_height = 80, 55, 620, 370
    series = [
        ("Exact entropy", "#2563eb", metrics["primary_k50"]["signals"]["exact_entropy"]),
        (
            "Semantic entropy",
            "#dc2626",
            metrics["primary_k50"]["signals"]["semantic_entropy"],
        ),
        (
            "Semantic entropy + no-answer",
            "#7c3aed",
            metrics["primary_k50"]["signals"]["semantic_entropy_inclusive"],
        ),
        (
            "Modal support",
            "#059669",
            metrics["primary_k50"]["signals"]["semantic_modal_support"],
        ),
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">K=50 uncertainty risk-coverage</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
    ]
    for value in range(0, 6):
        fraction = value / 5
        x = left + fraction * plot_width
        y = top + (1.0 - fraction) * plot_height
        elements.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="sans-serif" font-size="11">{fraction:.1f}</text>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{fraction:.1f}</text>',
            ]
        )
    for index, (label, color, signal) in enumerate(series):
        points = [
            (
                left + float(point["coverage"]) * plot_width,
                top + (1.0 - float(point["risk"])) * plot_height,
            )
            for point in signal["risk_coverage"]
        ]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        elements.append(
            f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, y in points:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>')
        legend_y = 68 + index * 22
        elements.extend(
            [
                f'<line x1="500" y1="{legend_y}" x2="530" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="538" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{label}</text>',
            ]
        )
    elements.extend(
        [
            f'<text x="{left + plot_width / 2}" y="480" text-anchor="middle" font-family="sans-serif" font-size="14">Coverage</text>',
            f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 {top + plot_height / 2})">Selective risk</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def build_provenance(raw: Path, manifest: Path, annotations: Path) -> dict[str, Any]:
    paths = {
        "raw": raw,
        "manifest": manifest,
        "annotations": annotations,
        "protocol": Path("docs/semantic_entropy_protocol.md"),
        "analysis_script": Path("scripts/analyze_semantic_entropy.py"),
    }
    return {
        "schema_version": 1,
        "analysis": "k50_empirical_semantic_entropy",
        "analysis_date": "2026-09-20",
        "synthetic_rollouts": 0,
        "new_model_inference": 0,
        "files_sha256": {name: sha256_file(path) for name, path in paths.items()},
    }


def serialized_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_or_validate(path: Path, expected: str, validate_only: bool) -> None:
    if validate_only:
        if not path.exists():
            raise ValueError(f"missing generated artifact: {path}")
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            raise ValueError(f"generated artifact is stale: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    raw_path = Path(args.raw)
    manifest_path = Path(args.manifest)
    annotations_path = Path(args.annotations)
    metrics, assignments = build_analysis(
        load_jsonl(raw_path),
        load_json(manifest_path),
        load_json(annotations_path),
        args.bootstrap_samples,
    )
    artifacts = {
        Path("results/semantic_entropy_metrics.json"): serialized_json(metrics),
        Path("data/processed/k50_semantic_cluster_assignments.json"): serialized_json(
            assignments
        ),
        Path("results/semantic_entropy_tasks.csv"): render_csv(metrics),
        Path("results/semantic_entropy_summary.md"): render_summary(metrics),
        Path("results/figures/semantic_entropy_risk_coverage.svg"): render_svg(metrics),
        Path("results/semantic_entropy_provenance.json"): serialized_json(
            build_provenance(raw_path, manifest_path, annotations_path)
        ),
    }
    for path, expected in artifacts.items():
        write_or_validate(path, expected, args.validate_only)
    mode = "validated" if args.validate_only else "wrote"
    primary = metrics["primary_k50"]
    print(
        f"semantic entropy {mode}: tasks={metrics['task_count']} "
        f"rollouts={metrics['rollout_count']} "
        f"semantic_auroc={primary['signals']['semantic_entropy']['auroc']:.3f}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"semantic entropy analysis failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
