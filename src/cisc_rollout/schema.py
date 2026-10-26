"""Typed records and JSON parsing for rollout logs."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class Evidence:
    path: str
    quote: str


@dataclass(frozen=True)
class FinalAnswer:
    final: str
    evidence: tuple[Evidence, ...] = ()


@dataclass(frozen=True)
class ToolObservation:
    action: str
    path: str
    output: str
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class Rollout:
    task_id: str
    seed: int
    final: FinalAnswer | None = None
    tool_observations: tuple[ToolObservation, ...] = ()
    parse_failed: bool = False
    tool_failed: bool = False
    generated_tokens: int | None = None
    wall_seconds: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return self.final is not None and not self.parse_failed

    @property
    def answer(self) -> str | None:
        return None if self.final is None else self.final.final

    @property
    def is_tool_grounded(self) -> bool:
        if self.final is None:
            return False
        observations_by_path: dict[str, list[str]] = {}
        for observation in self.tool_observations:
            if observation.ok:
                observations_by_path.setdefault(observation.path, []).append(observation.output)
        for evidence in self.final.evidence:
            quote = evidence.quote
            if len(quote) < 4:
                continue
            for output in observations_by_path.get(evidence.path, []):
                if quote in output:
                    return True
        return False


@dataclass(frozen=True)
class TaskItem:
    id: str
    split: str
    task_path: str
    question: str
    aliases: tuple[str, ...]


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first balanced JSON object from a model turn."""

    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    starts = [0] if stripped.startswith("{") else []
    starts.extend(index for index, char in enumerate(stripped) if char == "{")
    seen: set[int] = set()
    for start in starts:
        if start in seen:
            continue
        seen.add(start)
        try:
            value, _ = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("no balanced JSON object found")


parse_json_object = extract_json_object


def parse_turn(text: str) -> dict[str, Any]:
    """Parse a model turn and validate the top-level action/final shape."""

    obj = extract_json_object(text)
    if "final" in obj:
        if not isinstance(obj["final"], str):
            raise ValueError("final must be a string")
        evidence = obj.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("evidence must be a list")
        for entry in evidence:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("quote"), str):
                raise ValueError("each evidence entry must contain string path and quote")
        return obj
    if "action" in obj:
        if not isinstance(obj["action"], str):
            raise ValueError("action must be a string")
        return obj
    raise ValueError("turn must contain either action or final")


def parse_model_output(text: str) -> dict[str, Any]:
    """Compatibility wrapper for parsing one model turn."""

    return parse_turn(text)


def final_from_json(obj: dict[str, Any]) -> FinalAnswer:
    if "final" not in obj:
        raise ValueError("not a final answer object")
    evidence = tuple(Evidence(path=item["path"], quote=item["quote"]) for item in obj.get("evidence", []))
    return FinalAnswer(final=obj["final"], evidence=evidence)


def rollout_from_record(record: dict[str, Any]) -> Rollout:
    """Build a Rollout from a JSONL-safe dictionary.

    Accepted final forms:
    - {"final": "answer", "evidence": [{"path": "...", "quote": "..."}]}
    - {"final": {"final": "answer", "evidence": [...]}}
    """

    final_value = record.get("final")
    final: FinalAnswer | None
    if isinstance(final_value, str):
        final = final_from_json({"final": final_value, "evidence": record.get("evidence", [])})
    elif isinstance(final_value, dict):
        final = final_from_json(final_value)
    elif final_value is None:
        final = None
    else:
        raise ValueError("final must be a string, object, or null")

    observations = []
    for item in record.get("tool_observations", record.get("tool_results", [])):
        if not isinstance(item, dict):
            raise ValueError("tool observation must be an object")
        action = item.get("action", item.get("tool", ""))
        path = item.get("path", "")
        output = item.get("output", "")
        if not isinstance(action, str) or not isinstance(path, str) or not isinstance(output, str):
            raise ValueError("tool observation action/path/output must be strings")
        observations.append(
            ToolObservation(
                action=action,
                path=path,
                output=output,
                ok=bool(item.get("ok", True)),
                error=item.get("error") if isinstance(item.get("error"), str) else None,
            )
        )
    return Rollout(
        task_id=str(record["task_id"]),
        seed=int(record["seed"]),
        final=final,
        tool_observations=tuple(observations),
        parse_failed=bool(record.get("parse_failed", False)),
        tool_failed=bool(record.get("tool_failed", False)),
        generated_tokens=record.get("generated_tokens") if isinstance(record.get("generated_tokens"), int) else None,
        wall_seconds=record.get("wall_seconds") if isinstance(record.get("wall_seconds"), (int, float)) else None,
        raw=record,
    )


def load_rollouts_jsonl(path: str) -> dict[str, list[Rollout]]:
    grouped: dict[str, list[Rollout]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                rollout = rollout_from_record(record)
            except Exception as exc:
                raise ValueError(f"invalid rollout record at {path}:{line_number}: {exc}") from exc
            grouped.setdefault(rollout.task_id, []).append(rollout)
    return grouped


def load_manifest(path: str) -> list[TaskItem]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        TaskItem(
            id=item["id"],
            split=item["split"],
            task_path=item["task_path"],
            question=item["question"],
            aliases=tuple(item["aliases"]),
        )
        for item in data["items"]
    ]
