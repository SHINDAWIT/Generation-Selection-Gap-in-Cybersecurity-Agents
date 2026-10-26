#!/usr/bin/env python3
"""Curator-only export from the preserved research workspace; no inference.

Requires unpublished input files. Public readers use verify_public_results.py.
Never changes the source workspace or its historical provenance files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from cisc_rollout.normalize import answers_match, normalize_answer
from scripts.analyze_semantic_entropy import classify_semantic_answer
from scripts.verify_public_results import analyze, validate_ledger


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(source: Path) -> None:
    main = read(source / "data/task_manifest.json")
    annotations = read(source / "data/annotations/k50_semantic_clusters.json")
    items = {
        "development": [dict(item, gold_aliases=item["aliases"], source="CyBench")
                        for item in main["items"] if item["split"] == "main"],
        "holdout": [dict(item, source="CyBench")
                    for item in read(source / "data/holdout_prepared_items.json")["items"]],
        "external": read(source / "data/external_transfer_gold.json")["items"],
    }
    raw_paths = {"development": "data/raw/main_k50_rollouts.jsonl",
                 "holdout": "data/raw/holdout_k20_rollouts.jsonl",
                 "external": "data/raw/external_transfer_k20_rollouts.jsonl"}
    cohorts, rows, inputs = {}, [], []
    for cohort, relative in raw_paths.items():
        raw_path = source / relative
        original = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        by_id = {item["id"]: item for item in items[cohort]}
        k = 50 if cohort == "development" else 20
        cohorts[cohort] = {
            "task_ids": list(by_id), "source_by_task": {key: item["source"] for key, item in by_id.items()},
            "seeds": [1101 * index for index in range(1, k + 1)],
            "model": "Granite 4.2 8B Q4_K_M", "k": k,
            "prompt_versions": dict(Counter(row["prompt_version"] for row in original)),
            "tie_rule": "first_seed" if cohort == "development" else "abstain",
            "threshold": "K=25: absolute count >=15" if cohort == "development" else "unique mode / valid answers >=0.60",
            "grading": "strict gold-alias; development semantic labels are retrospective",
        }
        inputs.append({"path": relative, "sha256": sha256(raw_path), "records": len(original),
                       "redistributed": False})
        for row in original:
            if row.get("infrastructure_failed"):
                raise ValueError("Source infrastructure failure requires explicit export review")
            final = row.get("final")
            answer = final.get("final", "") if isinstance(final, dict) else (final or "")
            normalized = "" if row.get("parse_failed") else normalize_answer(answer)
            key = hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else None
            semantic = classify_semantic_answer(row["task_id"], normalized, annotations) if (
                normalized and cohort == "development") else None
            rows.append({
                "cohort": cohort, "task_id": row["task_id"], "source": by_id[row["task_id"]]["source"],
                "seed": row["seed"], "answer_key": key,
                "strict_correct": bool(normalized and answers_match(normalized, by_id[row["task_id"]]["gold_aliases"])),
                "semantic_cluster": semantic["cluster_id"] if semantic else None,
                "semantic_label": semantic["correctness"] if semantic else None,
            })
    validate_ledger(rows, cohorts)
    counts = analyze(rows)
    k5 = read(source / "results/main_metrics.json")
    k25 = read(source / "results/k25_metrics.json")
    sem = read(source / "results/semantic_entropy_metrics.json")["primary_k50"]
    reference = {"development_strict": [], "development_semantic_k50": {
        "tasks": 15, "oracle_correct": sem["semantic_oracle_tasks"],
        "modal_correct": sem["modal_correct"], "modal_incorrect": sem["modal_incorrect"],
        "modal_ambiguous": sem["modal_ambiguous"],
    }}
    for k, data in [(5, k5), (25, k25)]:
        reference["development_strict"].append({
            "k": k, "tasks": 15,
            "oracle_correct": round(data["selectors"]["oracle"]["overall_accuracy"] * 15),
            "modal_correct": round(data["selectors"]["majority"]["overall_accuracy"] * 15),
        })
    reference_paths = ["results/main_metrics.json", "results/k25_metrics.json", "results/k50_metrics.json",
                       "results/semantic_entropy_metrics.json", "data/task_manifest.json",
                       "data/holdout_prepared_items.json", "data/external_transfer_gold.json",
                       "data/annotations/k50_semantic_clusters.json"]
    for cohort, relative in [("holdout", "results/holdout_k20_metrics.json"),
                             ("external", "results/external_transfer_metrics.json")]:
        data = read(source / relative)
        selections = data["selectors"]
        before = selections["exact_modal_without_support_abstention"]
        after = selections["exact_majority_support_ge_0_60"]
        reference[cohort] = {
            "tasks": data["task_count"],
            "rollouts": 220 if cohort == "holdout" else 700,
            "single_correct": selections["single_seed_1101"]["correct"],
            "oracle_correct": selections["exact_oracle_pass_at_20"]["correct"],
            "before_threshold": {"correct": before["correct"], "incorrect": before["incorrect"],
                                 "abstained": before["abstained_on_tie_or_no_valid"]},
            "after_threshold": {key: after[key] for key in ("correct", "incorrect", "abstained")},
            "invalid_rollouts": round(data["invalid_rate"] * 220) if cohort == "holdout" else sum(
                task["invalid_count"] for task in data["tasks"]),
        }
        reference_paths.append(relative)
    if counts != reference:
        raise ValueError(f"Projection and stored aggregates disagree: {counts!r} vs {reference!r}")
    write(ROOT / "data/public/cohorts.json", cohorts)
    (ROOT / "data/public/rollout_answers.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    write(ROOT / "results/paper_results.json", {
        "schema_version": 1, "description": "Integer count projection of preserved metrics; no new inference",
        "scoring": "Strict primary; semantic K=50 is a separate retrospective diagnostic",
        "counts": reference,
        "references": [{"path": path, "sha256": sha256(source / path)} for path in reference_paths],
    })
    write(ROOT / "data/public/source_inputs.json", inputs)
    old = read(source / "results/k25_provenance.json")["harness_sha256"]
    changes = [{"path": path, "recorded_sha256": expected, "current_sha256": sha256(source / path)}
               for path, expected in old.items() if sha256(source / path) != expected]
    write(ROOT / "results/reference/k25_code_version_audit.json", {
        "description": "Historical provenance preserved; publication does not rewrite it as a match",
        "historical_provenance_sha256": sha256(source / "results/k25_provenance.json"),
        "changed_files": changes,
        "additional_modules_now_present": [str(p.relative_to(source)) for p in sorted((source / "src/cisc_rollout").glob("*.py"))
                                           if str(p.relative_to(source)) not in old],
    })
    print(f"Exported {len(rows)} source-derived records; integer counts match preserved metrics.")


def manifest(source: Path) -> None:
    files = {}
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
                                     for part in relative.parts) or str(relative) == "publication_manifest.json":
            continue
        origin = source / relative
        if str(relative) == "CISC-W_2026_paper__324.pdf":
            origin = source / "cursor-to-WSL/CISC-W_2026_paper__324 (2).pdf"
        preserved = origin.is_file() and sha256(origin) == sha256(path)
        files[str(relative)] = {"sha256": sha256(path), "bytes": path.stat().st_size,
                                "kind": "preserved_snapshot" if preserved else "publication_created"}
        if preserved:
            files[str(relative)]["source_path"] = str(origin.relative_to(source))
    write(ROOT / "publication_manifest.json", {
        "schema_version": 1, "paper_id": 324,
        "scope": "Current source snapshots and a derived public ledger; no claim of historical code identity",
        "files": files,
    })
    print(f"Manifest: {len(files)} files; original input hashes are in data/public/source_inputs.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    source = args.source_root.resolve()
    if source == ROOT:
        parser.error("source-root must be the separate research workspace")
    if not args.manifest_only:
        export(source)
    manifest(source)
