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
D-Cut adaptive-verify monkey patch installed for vLLM Ascend.
D-Cut adaptive verify ENABLED (...)
```
