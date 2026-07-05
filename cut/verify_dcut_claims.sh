#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

python3 - <<'PY'
from cut.verify_adaptive_controller import choose_query_lens_discrete
r = choose_query_lens_discrete(
    probs=[[0.9, 0.8], [0.5, 0.5]],
    base_batch_size=2,
    q_levels=[2, 4, 6],
    cost_lookup=lambda q: {2: 1.0, 4: 1.2, 6: 2.0}[q],
    max_draft_len=2,
)
assert r["draft_lens"] == [2, 0]
assert r["best_Q"] == 4
print("D-Cut controller smoke check passed")
PY
