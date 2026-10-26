"""Utilities for the CISC multi-rollout selection experiment."""

from .metrics import bootstrap_percentile_ci, evaluate_tasks
from .normalize import answers_match, normalize_answer
from .schema import (
    Evidence,
    FinalAnswer,
    Rollout,
    TaskItem,
    ToolObservation,
    load_manifest,
    load_rollouts_jsonl,
    parse_json_object,
    parse_model_output,
    rollout_from_record,
)
from .selectors import (
    Selection,
    agreement_abstention,
    normalized_majority,
    oracle_pass_at_k,
    select_all,
    single_first,
    tool_grounded_majority,
)
from .tools import ReadOnlyToolset, ToolError, ToolSandbox

__all__ = [
    "Evidence",
    "FinalAnswer",
    "ReadOnlyToolset",
    "Rollout",
    "Selection",
    "TaskItem",
    "ToolError",
    "ToolSandbox",
    "ToolObservation",
    "agreement_abstention",
    "answers_match",
    "bootstrap_percentile_ci",
    "evaluate_tasks",
    "load_manifest",
    "load_rollouts_jsonl",
    "normalize_answer",
    "normalized_majority",
    "oracle_pass_at_k",
    "parse_json_object",
    "parse_model_output",
    "rollout_from_record",
    "select_all",
    "single_first",
    "tool_grounded_majority",
]
