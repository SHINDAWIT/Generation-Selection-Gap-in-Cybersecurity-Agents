#!/usr/bin/env python3
"""Prepare the frozen static-artifact CyBench subset for Topic 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "task_manifest.json"
OUT_ROOT = REPO_ROOT / "data" / "benchmark"
PREPARED_PATH = REPO_ROOT / "data" / "prepared_items.json"

PUBLIC_ROOT_PRIORITY = ("release", "dist", "public", "html", "challenge", "env")
ARCHIVE_SUFFIXES = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"}
MAX_ARCHIVE_MEMBER_BYTES = 100 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 500 * 1024 * 1024
EXCLUDED_PARTS = {
    ".git",
    ".github",
    "__pycache__",
    "metadata",
    "solution",
    "solutions",
    "writeup",
    "writeups",
    "target",
    ".gradle",
}
EXCLUDED_FILENAMES = {
    "dockerfile",
    "docker-compose.yml",
    "compose.yml",
    "start_docker.sh",
    "stop_docker.sh",
    "start_script.sh",
    "stop_script.sh",
    "init_script.sh",
    "entrypoint.sh",
    "requirements.sh",
}
EXCLUDED_PREFIXES = ("readme",)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def normalize_answer(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def git_head(path: Path) -> str:
    import subprocess

    proc = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout.strip()


def is_safe_relative(path: PurePosixPath) -> bool:
    return not path.is_absolute() and ".." not in path.parts and bool(path.parts)


def is_excluded_rel(rel: PurePosixPath) -> bool:
    lower_parts = tuple(part.lower() for part in rel.parts)
    name = lower_parts[-1]
    stem = PurePosixPath(name).stem
    if any(part in EXCLUDED_PARTS for part in lower_parts):
        return True
    if name in EXCLUDED_FILENAMES:
        return True
    if any(name.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    if name.startswith("flag.") or stem == "flag":
        return True
    if name.endswith((".pyc", ".pyo")):
        return True
    return False


def candidate_roots(task_dir: Path) -> list[Path]:
    roots = [task_dir / name for name in PUBLIC_ROOT_PRIORITY if (task_dir / name).exists()]
    if not roots:
        raise RuntimeError(f"no public artifact root found in {task_dir}")
    first_priority = next(name for name in PUBLIC_ROOT_PRIORITY if (task_dir / name).exists())
    return [task_dir / first_priority]


def copy_artifacts(task_dir: Path, out_dir: Path) -> list[dict[str, str]]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    copied: list[Path] = []
    for root in candidate_roots(task_dir):
        if root.is_file():
            sources = [root]
        else:
            sources = sorted(
                path for path in root.rglob("*") if path.is_file() and not path.is_symlink()
            )
        for src in sources:
            try:
                src.resolve().relative_to(task_dir.resolve())
            except ValueError as exc:
                raise RuntimeError(f"artifact path escaped task root: {src}") from exc
            rel = PurePosixPath(src.relative_to(task_dir).as_posix())
            if not is_safe_relative(rel) or is_excluded_rel(rel):
                continue
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
            if src.suffix.lower() in ARCHIVE_SUFFIXES or "".join(src.suffixes[-2:]).lower() in {
                ".tar.gz",
                ".tar.bz2",
                ".tar.xz",
            }:
                copied.extend(extract_archive(dst, out_dir))

    if not copied:
        raise RuntimeError(f"no artifacts copied from {task_dir}")

    records = []
    for path in sorted(set(copied), key=lambda p: p.relative_to(out_dir).as_posix()):
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(out_dir).as_posix(),
                    "sha256": sha256_path(path),
                    "bytes": str(path.stat().st_size),
                }
            )
    return records


def archive_extract_dir(archive_path: Path, out_dir: Path) -> Path:
    rel = archive_path.relative_to(out_dir)
    safe_name = "__".join(rel.parts)
    return out_dir / "_extracted" / safe_name


def extract_archive(archive_path: Path, out_dir: Path) -> list[Path]:
    target_dir = archive_extract_dir(archive_path, out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    total_extracted_bytes = 0

    def write_member(member_name: str, source: BinaryIO, declared_size: int) -> None:
        nonlocal total_extracted_bytes
        rel = PurePosixPath(member_name)
        if not is_safe_relative(rel) or is_excluded_rel(rel):
            return
        if declared_size < 0 or declared_size > MAX_ARCHIVE_MEMBER_BYTES:
            raise RuntimeError(f"archive member exceeds size limit: {member_name}")
        if total_extracted_bytes + declared_size > MAX_ARCHIVE_TOTAL_BYTES:
            raise RuntimeError("archive exceeds cumulative extraction size limit")
        dst = target_dir / rel
        try:
            dst.resolve().relative_to(target_dir.resolve())
        except ValueError as exc:
            raise RuntimeError(f"archive path escaped target: {member_name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        try:
            with dst.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_ARCHIVE_MEMBER_BYTES:
                        raise RuntimeError(f"archive member exceeds size limit: {member_name}")
                    if total_extracted_bytes + written > MAX_ARCHIVE_TOTAL_BYTES:
                        raise RuntimeError("archive exceeds cumulative extraction size limit")
                    output.write(chunk)
        except Exception:
            dst.unlink(missing_ok=True)
            raise
        total_extracted_bytes += written
        extracted.append(dst)

    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path) as zf:
                for info in sorted(zf.infolist(), key=lambda i: i.filename):
                    if info.is_dir():
                        continue
                    unix_mode = (info.external_attr >> 16) & 0xFFFF
                    if unix_mode and stat.S_ISLNK(unix_mode):
                        continue
                    with zf.open(info, "r") as source:
                        write_member(info.filename, source, info.file_size)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path) as tf:
                members = sorted(tf.getmembers(), key=lambda m: m.name)
                for member in members:
                    if not member.isfile():
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    with fh:
                        write_member(member.name, fh, member.size)
    except (zipfile.BadZipFile, tarfile.TarError, EOFError):
        return []
    return extracted


def prepare_item(cybench_benchmark: Path, item: dict[str, Any]) -> dict[str, Any]:
    task_dir = cybench_benchmark / item["task_path"]
    metadata_path = task_dir / "metadata" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    metadata = load_json(metadata_path)

    matching = [sub for sub in metadata["subtasks"] if sub.get("question") == item["question"]]
    if len(matching) != 1:
        raise RuntimeError(f"{item['id']}: expected one matching subtask, found {len(matching)}")
    subtask = matching[0]

    upstream_gold = str(subtask["answer"])
    aliases = [str(alias) for alias in item["aliases"]]
    if normalize_answer(upstream_gold) not in {normalize_answer(alias) for alias in aliases}:
        raise RuntimeError(f"{item['id']}: manifest aliases do not include upstream gold {upstream_gold!r}")

    artifact_dir = OUT_ROOT / item["id"]
    artifacts = copy_artifacts(task_dir, artifact_dir)

    return {
        "id": item["id"],
        "split": item["split"],
        "source_task_path": item["task_path"],
        "artifact_dir": artifact_dir.relative_to(REPO_ROOT).as_posix(),
        "prompt": metadata.get("hard_prompt") or metadata.get("easy_prompt") or "",
        "question": item["question"],
        "answer_format": subtask.get("answer_format", ""),
        "gold_aliases": aliases,
        "upstream_gold_sha256": hashlib.sha256(upstream_gold.encode("utf-8")).hexdigest(),
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cybench-root", required=True, type=Path)
    parser.add_argument("--allow-commit-mismatch", action="store_true")
    args = parser.parse_args()

    cybench_root = args.cybench_root.resolve()
    manifest = load_json(MANIFEST_PATH)
    expected_commit = manifest["source"]["commit"]
    actual_commit = git_head(cybench_root)
    if actual_commit != expected_commit and not args.allow_commit_mismatch:
        raise RuntimeError(f"CyBench commit mismatch: expected {expected_commit}, got {actual_commit}")

    benchmark_root = cybench_root / "benchmark"
    prepared = [prepare_item(benchmark_root, item) for item in manifest["items"]]
    output = {
        "schema_version": 1,
        "source": {
            "name": manifest["source"]["name"],
            "repository": manifest["source"]["repository"],
            "commit": expected_commit,
            "local_commit": actual_commit,
            "commit_verified": actual_commit == expected_commit,
        },
        "model_facing_fields": ["id", "prompt", "question", "answer_format", "artifact_dir"],
        "excluded_model_fields": ["gold_aliases", "upstream_gold_sha256"],
        "items": prepared,
    }
    dump_json(PREPARED_PATH, output)
    print(f"prepared {len(prepared)} items")
    print(f"wrote {PREPARED_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
