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
- `apply_adaptive_lengths` defaults to `true` when a D-Cut config is loaded, so
  adaptive planning and draft-length application are enabled by default for D-Cut
  benchmark runs. Set it to `false` only when you want logging/patch smoke checks
  without changing decoding behavior.
- `min_prefix_prob` filters low-confidence draft prefixes out of the adaptive
  planner. Increase it if a small number of requests repeatedly run to the
  generation limit with low acceptance.
- `min_adaptive_draft_len` defaults to `2` and is applied as a runtime floor
  when D-Cut rewrites scheduled draft tokens. This avoids pathological `0`/`1`
  token verifier segments that can destabilize high-concurrency Ascend/GDN
  paths. Lower it only when explicitly testing aggressive truncation.
- `uniform_adaptive_lengths` defaults to `true`, so each batch applies one
  adaptive draft length to all requests. Keep this on for Ascend runs; fully
  per-request variable draft lengths are experimental and can disturb
  spec-decode metadata on some kernels.
- `mutate_scheduler_output` defaults to `true`, so `apply_adaptive_lengths=true`
  actually rewrites scheduled draft lengths and D-Cut is active. Set it to
  `false` only for observe-only/debug runs where you want adaptive plans and
  logs without changing vLLM/Ascend scheduler outputs.
- `log_concurrency_interval_s` controls periodic server-side INFO logs for
  actual runner concurrency (`active_reqs`, `scheduled_reqs`, `spec_reqs`).
  Set it to `0` to disable these logs. If no D-Cut config is loaded,
  `VLLM_DCUT_LOG_CONCURRENCY_INTERVAL_S` is used and defaults to `5.0`.
- `log_runtime_events` controls high-frequency runtime `[DCUT]` logs such as
  `D-Cut plan`, `D-Cut apply`, and `D-Cut concurrency`. It defaults to
  `false`; keep it off for throughput benchmarks and set it to `true` only
  for short validation runs. Startup, dormant, and patch/install status logs
  are still emitted.

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

For a live server, temporarily set `log_runtime_events=true` and check logs for (`[DCUT]` is printed directly to stdout in addition to logger output):

```text
[DCUT] D-Cut adaptive-verify plugin install requested (VLLM_PLUGINS=ascend,dcut_adaptive_verify, config_env=/tmp/dcut_config.json).
[DCUT] D-Cut adaptive-verify delayed import hook installed.
[DCUT] D-Cut adaptive-verify plugin installed for vLLM Ascend (patches are applied lazily after Ascend runner modules load).
[DCUT] D-Cut adaptive-verify patched NPUModelRunner.
[DCUT] D-Cut adaptive-verify patched AscendSpecDecodeBaseProposer.
[DCUT] D-Cut adaptive verify ENABLED (config=...)
[DCUT] D-Cut adaptive verify ACTIVE: computed first adaptive draft-length plan (...)
[DCUT] D-Cut plan: batch=... verify_query_tokens=... query_len_per_req=... draft_lens_total=... draft_lens_avg=... draft_lens_min=... draft_lens_max=... draft_lens_hist=...
[DCUT] D-Cut adaptive verify ACTIVE: truncated scheduled draft tokens for the first time (...)
[DCUT] D-Cut concurrency: active_reqs=... scheduled_reqs=... spec_reqs=... total_scheduled_tokens=... max_scheduled_tokens_per_req=...
```

The install-hook lines prove vLLM discovered the plugin without eagerly importing
Ascend runner modules during CLI setup. The `patched ...` lines appear once the
Ascend modules load normally. The `ENABLED` line proves the runner accepted the
config and speculative method. The `ACTIVE`, `plan`, and `apply` lines appear after traffic starts and
prove D-Cut is actually computing and applying adaptive verifier lengths. Set
`log_runtime_events` should normally remain `false` for throughput benchmarks; enable it only for short validation runs if these per-step logs are needed.

Useful log checks:

```bash
rg -i "\[DCUT\]|D-Cut adaptive|dcut" /path/to/server.log | tail -80
```

If the plugin is installed but inactive, the log explains why, for example a
missing config env var, `enabled=false`, unsupported speculative method, or async
scheduling warning.
