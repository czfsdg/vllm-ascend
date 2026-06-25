# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence


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
    """Choose per-request draft lengths with a batch-wide D-Cut top-K scan.

    The verifier always processes one anchor token per active request.  For a
    candidate total query length ``Q``, the extra draft-token budget is
    ``Q - base_batch_size``.  D-Cut ranks marginal accepted-token gains by the
    prefix product of selected draft-token probabilities and maximizes
    ``(base_batch_size + accepted_gain) / verifier_cost(Q)``.
    """
    num_reqs = len(probs)
    if max_draft_len <= 0 or num_reqs == 0:
        return {
            "draft_lens": [0] * num_reqs,
            "best_Q": max(1, base_batch_size),
            "best_score": 0.0,
        }

    base_batch_size = max(1, int(base_batch_size))
    max_query_len = base_batch_size * (max_draft_len + 1)
    q_levels = sorted({max(base_batch_size, min(int(q), max_query_len)) for q in q_levels})
    if not q_levels:
        q_levels = [base_batch_size, max_query_len]

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
        score = (base_batch_size + accepted_gain) / cost
        if score > best_score:
            best_score = score
            best_k = k
            best_q = q

    draft_lens = [0] * num_reqs
    for _, req_idx in gains[:best_k]:
        draft_lens[req_idx] += 1
    draft_lens = [min(max_draft_len, draft_len) for draft_len in draft_lens]
    return {"draft_lens": draft_lens, "best_Q": best_q, "best_score": best_score}


def make_cost_lookup(
    cost_table: Mapping[str, float] | None,
    base_batch_size: int,
) -> Callable[[int], float]:
    """Build a verifier cost lookup for a batch from JSON cost-table data.

    Keys can be either ``"Q"`` or ``"bs,Q"``.  The latter lets one config carry
    multiple profiled batch-size rows; this helper uses the first row whose
    batch key is greater than or equal to the active batch size.
    """
    if not cost_table:
        return lambda q: float(q)

    simple: dict[int, float] = {}
    keyed: dict[tuple[int, int], float] = {}
    for raw_key, raw_cost in cost_table.items():
        cost = float(raw_cost)
        parts = [part.strip() for part in str(raw_key).split(",")]
        if len(parts) == 1:
            simple[int(parts[0])] = cost
        elif len(parts) == 2:
            keyed[(int(parts[0]), int(parts[1]))] = cost
        else:
            raise ValueError(f"Invalid D-Cut cost-table key: {raw_key!r}")

    candidate_bs = sorted({bs for bs, _ in keyed if bs >= base_batch_size})
    selected_bs = candidate_bs[0] if candidate_bs else None

    def lookup(q: int) -> float:
        if selected_bs is not None and (selected_bs, q) in keyed:
            return keyed[(selected_bs, q)]
        if q in simple:
            return simple[q]
        # Fall back to the nearest profiled Q on the chosen row/simple axis.
        row = {row_q: cost for (bs, row_q), cost in keyed.items() if bs == selected_bs}
        candidates = row or simple
        if candidates:
            nearest_q = min(candidates, key=lambda profiled_q: abs(profiled_q - q))
            return candidates[nearest_q]
        return float(q)

    return lookup
