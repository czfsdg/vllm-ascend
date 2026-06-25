# D-Cut adaptive verifier step-length — vLLM Ascend plugin

This directory is a self-contained `vllm.general_plugins` plugin. It does not
edit vLLM or vLLM Ascend source files at install time.

## Install

```bash
cd /path/to/vllm-ascend/dcut
pip install -e .
```

## Enable

```bash
cp verify_adaptive_config.example.json /tmp/dcut_config.json
export VLLM_DCUT_CONFIG=/tmp/dcut_config.json
export VLLM_PLUGINS=ascend,dcut_adaptive_verify
```

`VLLM_ASCEND_DCUT_CONFIG` is also accepted as an alias for
`VLLM_DCUT_CONFIG`. If neither config env var is set, the plugin loads but stays
dormant.

## Scope

- Active only for DFlash, or `method=draft_model` with `parallel_drafting=true`.
- MTP is intentionally unsupported and remains dormant.
- Async scheduling is not adapted; if enabled, the plugin logs a warning.
- If `cost_table` is provided in the JSON config, D-Cut uses those profiled
  verifier costs. Keys may be `"Q"` or `"bs,Q"`; batch-keyed rows use the
  smallest profiled batch size greater than or equal to the active batch.
  Without `cost_table`, it falls back to a monotonic synthetic `cost=Q` model.
- `apply_adaptive_lengths` defaults to `false`. In this safe mode the plugin
  computes and logs D-Cut plans but does not post-hoc truncate already scheduled
  draft tokens, because scheduler-side token accounting must remain identical
  between engine and worker processes for correctness.

## Smoke checks

```bash
cd /tmp
python3 - <<'PY'
from importlib.metadata import entry_points
plugins = [ep.name for ep in entry_points(group="vllm.general_plugins")]
print(plugins)
assert "dcut_adaptive_verify" in plugins
PY
```

For a live server, check logs for:

```text
D-Cut adaptive-verify plugin install requested (VLLM_PLUGINS=ascend,dcut_adaptive_verify, config_env=/tmp/dcut_config.json).
D-Cut adaptive-verify delayed import hook installed.
D-Cut adaptive-verify plugin installed for vLLM Ascend (patches are applied lazily after Ascend runner modules load).
D-Cut adaptive-verify patched NPUModelRunner.
D-Cut adaptive-verify patched AscendSpecDecodeBaseProposer.
D-Cut adaptive verify ENABLED (config=...)
D-Cut adaptive verify ACTIVE: computed first adaptive draft-length plan (...)
D-Cut adaptive verify ACTIVE: truncated scheduled draft tokens for the first time (...)
```

The install-hook lines prove vLLM discovered the plugin without eagerly importing
Ascend runner modules during CLI setup. The `patched ...` lines appear once the
Ascend modules load normally. The `ENABLED` line proves the runner accepted the
config and speculative method. The `ACTIVE` lines appear after traffic starts and
prove D-Cut is actually computing and applying adaptive verifier lengths.

Useful log checks:

```bash
rg -i "D-Cut adaptive|dcut" /path/to/server.log | tail -80
```

If the plugin is installed but inactive, the log explains why, for example a
missing config env var, `enabled=false`, unsupported speculative method, or async
scheduling warning.
