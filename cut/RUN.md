# D-Cut adaptive verifier step-length

This directory mirrors the file layout from the GPU D-Cut implementation at
`Bensong0506/vllm` branch `feat/dcut-adaptive-verify`, path `dcut/`.

The copied plugin exposes a `vllm.general_plugins` entry point named
`dcut_adaptive_verify` and a pure controller algorithm for adaptive verifier
step-length experiments.

## Quick algorithm check

```bash
python3 - <<'PY'
from cut.verify_adaptive_controller import choose_query_lens_discrete
r = choose_query_lens_discrete(
    probs=[[0.9, 0.8], [0.5, 0.5]],
    base_batch_size=2,
    q_levels=[2, 4, 6],
    cost_lookup=lambda q: {2: 1.0, 4: 1.2, 6: 2.0}[q],
    max_draft_len=2,
)
print(r)
assert r["draft_lens"] == [2, 0]
assert r["best_Q"] == 4
PY
```
