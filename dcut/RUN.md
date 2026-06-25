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

By default the example config enables active truncation with
`"apply_truncation": true` and prints one adaptive plan every 50 plans via
`"log_every_n_plans": 50`. Set `"apply_truncation": false` to run in
observe-only mode without changing verifier inputs.

The plan log includes the verifier token budget (`verifier_tokens`) and the
applied draft prefix lengths (`draft_lens`). Active truncation applies a
batch-uniform draft prefix length derived from the selected verifier budget,
which avoids ragged per-request verifier shapes on Ascend DFlash.

## Scope

- Active only for DFlash, or `method=draft_model` with `parallel_drafting=true`.
- MTP is intentionally unsupported and remains dormant.
- Async scheduling is not adapted; if enabled, the plugin logs a warning.
- The first Ascend plugin version uses a monotonic synthetic cost model for the
  verifier query levels. Replace `cost_lookup` in `dcut/monkeypatch.py` with a
  hardware-profiled table when NPU profiling data is available.

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
# If log_every_n_plans > 0:
D-Cut adaptive verify PLAN: count=... batch=... verifier_tokens=... draft_lens=...
D-Cut adaptive verify observe-only mode: computed plans are not applied.
# If apply_truncation=true:
D-Cut adaptive verify ACTIVE: truncated scheduled draft tokens for the first time (...)
D-Cut adaptive verify raised a truncation plan to keep scheduled tokens >= previously accepted tokens.
```

The install-hook lines prove vLLM discovered the plugin without eagerly importing
Ascend runner modules during CLI setup. The `patched ...` lines appear once the
Ascend modules load normally. The `ENABLED` line proves the runner accepted the
config and speculative method. In the default observe-only mode, the observe-only
line proves D-Cut computed a plan but intentionally did not change verifier
inputs. If `apply_truncation=true`, the `ACTIVE` truncation line proves D-Cut is
actually applying adaptive verifier lengths.

Useful log checks:

```bash
grep -Ei "D-Cut adaptive|dcut" /path/to/server.log | tail -80
```

If the plugin is installed but inactive, the log explains why, for example a
missing config env var, `enabled=false`, unsupported speculative method, or async
scheduling warning.
