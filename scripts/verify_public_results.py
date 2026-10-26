#!/usr/bin/env python3
"""Recount the public answer ledger without a model or benchmark answer file.

Added for publication. Correctness labels are inherited offline annotations,
not independently regraded from original answers by this utility.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def task_result(rows: list[dict], *, tie_abstains: bool) -> dict:
    ordered = sorted(rows, key=lambda row: row["seed"])
    valid = [row for row in ordered if row["answer_key"] is not None]
    counts = Counter(row["answer_key"] for row in valid)
    top = max(counts.values(), default=0)
    winners = {key for key, count in counts.items() if count == top}
    # Iterating the seed order preserves the historical development tie rule.
    chosen = next((row for row in valid if row["answer_key"] in winners), None)
    answered = chosen is not None and (not tie_abstains or len(winners) == 1)
    support = top / len(valid) if valid else 0.0
    policy_answered = answered and support >= 0.60
    return {
        "single_correct": ordered[0]["strict_correct"],
        "oracle_correct": any(row["strict_correct"] for row in ordered),
        "modal_correct": bool(answered and chosen["strict_correct"]),
        "modal_answered": answered,
        "policy_correct": bool(policy_answered and chosen["strict_correct"]),
        "policy_answered": policy_answered,
        "modal_semantic_label": chosen.get("semantic_label") if answered else None,
        "semantic_oracle": any(row.get("semantic_label") == "correct" for row in valid),
        "invalid": len(ordered) - len(valid),
        "modal_count": top,
        "support": support,
    }


def analyze(rows: list[dict]) -> dict:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["cohort"]][row["task_id"]].append(row)
    development = []
    for k in (5, 25):
        outcomes = [task_result(sorted(task, key=lambda r: r["seed"])[:k], tie_abstains=False)
                    for task in grouped["development"].values()]
        development.append({"k": k, "tasks": len(outcomes),
                            "oracle_correct": sum(x["oracle_correct"] for x in outcomes),
                            "modal_correct": sum(x["modal_correct"] for x in outcomes)})
    semantic_tasks = [task_result(task, tie_abstains=False)
                      for task in grouped["development"].values()]
    labels = Counter(task["modal_semantic_label"] for task in semantic_tasks)
    result = {"development_strict": development,
              "development_semantic_k50": {
                  "tasks": len(semantic_tasks),
                  "oracle_correct": sum(task["semantic_oracle"] for task in semantic_tasks),
                  "modal_correct": labels["correct"],
                  "modal_incorrect": labels["incorrect"],
                  "modal_ambiguous": labels["ambiguous"],
              }}
    for cohort in ("holdout", "external"):
        tasks = [task_result(task, tie_abstains=True) for task in grouped[cohort].values()]
        modal_answered = sum(task["modal_answered"] for task in tasks)
        modal_correct = sum(task["modal_correct"] for task in tasks)
        answered = sum(task["policy_answered"] for task in tasks)
        correct = sum(task["policy_correct"] for task in tasks)
        result[cohort] = {
            "tasks": len(tasks), "rollouts": sum(len(task) for task in grouped[cohort].values()),
            "single_correct": sum(task["single_correct"] for task in tasks),
            "oracle_correct": sum(task["oracle_correct"] for task in tasks),
            "before_threshold": {"correct": modal_correct, "incorrect": modal_answered - modal_correct,
                                 "abstained": len(tasks) - modal_answered},
            "after_threshold": {"correct": correct, "incorrect": answered - correct,
                                "abstained": len(tasks) - answered},
            "invalid_rollouts": sum(task["invalid"] for task in tasks),
        }
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_ledger(rows: list[dict], cohorts: dict) -> None:
    keys = set()
    labels = {}
    expected_fields = {"cohort", "task_id", "source", "seed", "answer_key", "strict_correct",
                       "semantic_cluster", "semantic_label"}
    for row in rows:
        require(set(row) == expected_fields, "unexpected ledger fields")
        cohort = row["cohort"]
        require(cohort in cohorts, "unknown cohort")
        config = cohorts[cohort]
        require(row["task_id"] in config["task_ids"], "unknown parent task")
        require(row["source"] == config["source_by_task"][row["task_id"]], "source mismatch")
        require(type(row["seed"]) is int and row["seed"] in config["seeds"], "unexpected seed")
        key = (cohort, row["task_id"], row["seed"])
        require(key not in keys, "duplicate parent/seed record")
        keys.add(key)
        require(type(row["strict_correct"]) is bool, "invalid correctness label")
        answer = row["answer_key"]
        if answer is None:
            require(row["strict_correct"] is False and row["semantic_cluster"] is None
                    and row["semantic_label"] is None, "invalid answer carries a label")
        else:
            require(isinstance(answer, str) and len(answer) == 64
                    and all(c in "0123456789abcdef" for c in answer), "malformed answer key")
            if cohort == "development":
                require(isinstance(row["semantic_cluster"], str) and bool(row["semantic_cluster"])
                        and row["semantic_label"] in {"correct", "incorrect", "ambiguous"},
                        "missing development semantic annotation")
            else:
                require(row["semantic_cluster"] is None and row["semantic_label"] is None,
                        "K=20 cohort has retrospective semantic labels")
            label_key = (cohort, row["task_id"], answer)
            value = (row["strict_correct"], row["semantic_cluster"], row["semantic_label"])
            require(label_key not in labels or labels[label_key] == value, "inconsistent answer labels")
            labels[label_key] = value
    expected = {(name, task_id, seed) for name, spec in cohorts.items()
                for task_id in spec["task_ids"] for seed in spec["seeds"]}
    require(keys == expected, "incomplete rollout grid")


def verify_manifest(root: Path) -> int:
    manifest = json.loads((root / "publication_manifest.json").read_text(encoding="utf-8"))
    tracked = manifest["files"]
    for relative, expected in tracked.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file(), f"missing file: {relative}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"],
                f"file hash mismatch: {relative}")
        require(path.stat().st_size == expected["bytes"], f"file size mismatch: {relative}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    return len(tracked)


def main() -> int:
    try:
        count = verify_manifest(ROOT)
        cohorts = json.loads((ROOT / "data/public/cohorts.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (ROOT / "data/public/rollout_answers.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()]
        validate_ledger(rows, cohorts)
        actual = analyze(rows)
        reference = json.loads((ROOT / "results/paper_results.json").read_text(encoding="utf-8"))
        require(actual == reference["counts"], "recount differs from paper/reference counts")
        motp = [row for row in rows if row["task_id"] == "main-motp-operator"]
        require(len(motp) == 50 and sum(row["strict_correct"] for row in motp) == 1
                and sorted(Counter(row["answer_key"] for row in motp).values()) == [1, 49],
                "MOTP minority-correct example mismatch")
        print(f"PASS: {len(rows)} rollout records, 61 parent tasks, complete grids; {count} file hashes.")
        print("PASS: strict K=5/25, retrospective semantic K=50, K=20 outcomes and MOTP 49:1.")
        print("Scope: recounts inherited answer labels; does not regrade source answers or run inference.")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Public verification failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
