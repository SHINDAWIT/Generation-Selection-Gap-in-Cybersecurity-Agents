"""Offline selectors for k-rollout answer sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .normalize import answers_match, normalize_answer
from .schema import Rollout


@dataclass(frozen=True)
class Selection:
    selector: str
    answer: str | None
    normalized_answer: str | None
    seed: int | None
    support: int
    considered: int
    abstained: bool
    correct: bool | None = None
    reason: str = ""


def _valid_rollouts(rollouts: Iterable[Rollout]) -> list[Rollout]:
    return [rollout for rollout in sorted(rollouts, key=lambda item: item.seed) if rollout.valid and rollout.answer]


def _with_correct(selection: Selection, aliases: tuple[str, ...] | list[str] | None) -> Selection:
    if aliases is None or selection.abstained or selection.answer is None:
        return selection
    return Selection(
        selector=selection.selector,
        answer=selection.answer,
        normalized_answer=selection.normalized_answer,
        seed=selection.seed,
        support=selection.support,
        considered=selection.considered,
        abstained=selection.abstained,
        correct=answers_match(selection.answer, aliases),
        reason=selection.reason,
    )


def single_first(rollouts: Iterable[Rollout], aliases: tuple[str, ...] | list[str] | None = None) -> Selection:
    valid = _valid_rollouts(rollouts)
    if not valid:
        return Selection("single", None, None, None, 0, 0, True, None, "no valid rollout")
    chosen = valid[0]
    return _with_correct(
        Selection("single", chosen.answer, normalize_answer(chosen.answer), chosen.seed, 1, len(valid), False),
        aliases,
    )


def _majority(name: str, rollouts: list[Rollout], aliases: tuple[str, ...] | list[str] | None = None) -> Selection:
    if not rollouts:
        return Selection(name, None, None, None, 0, 0, True, None, "no eligible rollout")
    groups: dict[str, list[Rollout]] = {}
    for rollout in rollouts:
        normalized = normalize_answer(rollout.answer)
        if normalized:
            groups.setdefault(normalized, []).append(rollout)
    if not groups:
        return Selection(name, None, None, None, 0, len(rollouts), True, None, "no non-empty answer")
    winner_norm, winner_group = min(
        groups.items(),
        key=lambda item: (-len(item[1]), min(rollout.seed for rollout in item[1])),
    )
    chosen = min(winner_group, key=lambda rollout: rollout.seed)
    return _with_correct(
        Selection(name, chosen.answer, winner_norm, chosen.seed, len(winner_group), len(rollouts), False),
        aliases,
    )


def normalized_majority(rollouts: Iterable[Rollout], aliases: tuple[str, ...] | list[str] | None = None) -> Selection:
    return _majority("majority", _valid_rollouts(rollouts), aliases)


def tool_grounded_majority(rollouts: Iterable[Rollout], aliases: tuple[str, ...] | list[str] | None = None) -> Selection:
    grounded = [rollout for rollout in _valid_rollouts(rollouts) if rollout.is_tool_grounded]
    return _majority("tool_grounded_majority", grounded, aliases)


def agreement_abstention(
    rollouts: Iterable[Rollout],
    aliases: tuple[str, ...] | list[str] | None = None,
    threshold: int = 3,
) -> Selection:
    selection = normalized_majority(rollouts, aliases)
    if selection.abstained:
        return Selection("agreement_abstention", None, None, None, 0, selection.considered, True, None, selection.reason)
    if selection.support < threshold:
        return Selection(
            "agreement_abstention",
            None,
            None,
            None,
            selection.support,
            selection.considered,
            True,
            None,
            f"support below threshold {threshold}",
        )
    return Selection(
        "agreement_abstention",
        selection.answer,
        selection.normalized_answer,
        selection.seed,
        selection.support,
        selection.considered,
        False,
        selection.correct,
        "",
    )


def oracle_pass_at_k(rollouts: Iterable[Rollout], aliases: tuple[str, ...] | list[str]) -> Selection:
    valid = _valid_rollouts(rollouts)
    for rollout in valid:
        if answers_match(rollout.answer, aliases):
            return Selection("oracle", rollout.answer, normalize_answer(rollout.answer), rollout.seed, 1, len(valid), False, True)
    return Selection("oracle", None, None, None, 0, len(valid), True, False, "no rollout matched gold aliases")


SelectorFn = Callable[[Iterable[Rollout], tuple[str, ...] | list[str]], Selection]


def select_all(
    rollouts: Iterable[Rollout],
    aliases: tuple[str, ...] | list[str],
    agreement_threshold: int = 3,
) -> dict[str, Selection]:
    items = list(rollouts)
    return {
        "single": single_first(items, aliases),
        "majority": normalized_majority(items, aliases),
        "tool_grounded_majority": tool_grounded_majority(items, aliases),
        "agreement_abstention": agreement_abstention(
            items, aliases, threshold=agreement_threshold
        ),
        "oracle": oracle_pass_at_k(items, aliases),
    }
