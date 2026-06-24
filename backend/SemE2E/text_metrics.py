from __future__ import annotations

import re
from collections.abc import Sequence


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: Sequence[object], hypothesis: Sequence[object]) -> int:
    prev = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        curr = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ref_item == hyp_item else 1),
                )
            )
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)
