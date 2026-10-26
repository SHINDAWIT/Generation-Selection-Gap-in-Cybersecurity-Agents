"""Answer normalization fixed by the predeclared protocol."""

from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.DOTALL)
_SURROUNDING_MARKUP = "`* \t\r\n"
_SURROUNDING_QUOTES = " \t\r\n\"'“”‘’"


def normalize_answer(value: object) -> str:
    """Normalize an answer conservatively for within-task comparison.

    The function intentionally avoids semantic rewrites. It folds Unicode,
    case, surrounding Markdown/quotes, repeated whitespace, and punctuation
    that appears only at the answer boundary. Internal punctuation is retained
    because it is often meaningful in cybersecurity answers.
    """

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _FENCE_RE.sub("", text.strip())
    text = text.strip(_SURROUNDING_MARKUP)
    text = text.strip(_SURROUNDING_QUOTES)
    text = _SPACE_RE.sub(" ", text)
    return text.casefold()


def normalized_aliases(aliases: list[str] | tuple[str, ...]) -> set[str]:
    return {norm for alias in aliases if (norm := normalize_answer(alias))}


def answers_match(answer: object, aliases: list[str] | tuple[str, ...]) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and normalized in normalized_aliases(aliases)
