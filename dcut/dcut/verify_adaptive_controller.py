# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence


def _prefix_scores(probs: Sequence[float], max_draft_len: int) -> list[float]:
    scores: list[float] = []
    prefix = 1.0
    for prob in list(probs)[:max_draft_len]:
        prefix *= max(0.0, min(1.0, float(prob)))
        scores.append(prefix)
    return scores


def choose_query_lens_discrete(
    probs: Sequence[Sequence[float]],
    base_batch_size: int,
    q_levels: Sequence[int],
    cost_lookup: Callable[[int], float],
    max_draft_len: int,
) -> dict[str, object]:
    """Choose per-request draft lengths with a batch-wide D-Cut top-K scan."""
    if max_draft_len <= 0 or not probs:
        return {"draft_lens": [0] * len(probs), "best_Q": base_batch_size, "best_score": 0.0}

    q_levels = sorted({int(q) for q in q_levels if int(q) >= 1})
    if not q_levels:
        q_levels = [base_batch_size, base_batch_size + max_draft_len]

    gains: list[tuple[float, int]] = []
    for req_idx, req_probs in enumerate(probs):
        for score in _prefix_scores(req_probs, max_draft_len):
            gains.append((score, req_idx))
    gains.sort(reverse=True, key=lambda item: item[0])

    best_score = float("-inf")
    best_k = 0
    best_q = base_batch_size
    for q in q_levels:
        draft_budget = max(0, q - base_batch_size)
        k = min(draft_budget, len(gains))
        accepted_gain = sum(score for score, _ in gains[:k])
        cost = max(float(cost_lookup(q)), 1e-9)
        score = accepted_gain / cost
        if score > best_score:
            best_score = score
            best_k = k
            best_q = q

    draft_lens = [0] * len(probs)
    for _, req_idx in gains[:best_k]:
        draft_lens[req_idx] += 1
    draft_lens = [min(max_draft_len, draft_len) for draft_len in draft_lens]
    return {"draft_lens": draft_lens, "best_Q": best_q, "best_score": best_score}
