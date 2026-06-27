# D-Cut adaptive verifier plugin

D-Cut is an optional standalone plugin for Ascend speculative decoding. It adapts
verifier draft length per request by profiling verifier cost and using the
selected draft-token probabilities from the drafter.

## Install only the plugin

If your `vllm-ascend` is already installed and you cannot reinstall it, install
only this plugin from the `dcut/` directory:

```bash
cd /path/to/vllm-ascend/dcut
pip install -e .
```

This installs the `vllm-ascend-dcut` package and exposes both `dcut` and
`dcut_adaptive_verify` vLLM general-plugin entrypoints. The package provides both
`register()` and `install()` entrypoint functions for compatibility with existing
D-Cut launch scripts. It does **not** reinstall `vllm-ascend`.

## Enable in the startup script

D-Cut is disabled by default. Add the enable flag before starting vLLM. The
`DCUT_*` variables avoid vLLM's unknown-environment-variable warning:

```bash
export DCUT_ENABLE=1
vllm serve <model> --speculative-config '<your speculative config>'
```

`VLLM_ASCEND_ENABLE_DCUT=1` is also accepted for compatibility.

If your environment restricts vLLM plugins through `VLLM_PLUGINS`, include either
`dcut` or `dcut_adaptive_verify` together with the existing Ascend plugin entries.
For example, `VLLM_PLUGINS=ascend` is not enough because it filters out D-Cut:

```bash
export VLLM_PLUGINS=ascend,dcut_adaptive_verify
```

In normal vLLM plugin discovery, setting `VLLM_PLUGINS` is not required.

## Optional config

When `VLLM_ASCEND_ENABLE_DCUT=1` and no config is provided, built-in defaults are
used. To override profiling levels or dump the cost table:

```bash
cat > /tmp/dcut_config.json <<'JSON'
{
  "min_warmup_batch_size": 2,
  "max_warmup_batch_size": 64,
  "query_len_step_per_req": 2,
  "min_query_len_per_req": 2,
  "n_warmup_iters": 3,
  "n_measure_iters": 5,
  "cost_table_dump_path": "/tmp/dcut_cost_table.json"
}
JSON

export DCUT_ENABLE=1
export DCUT_CONFIG=/tmp/dcut_config.json
vllm serve <model> --speculative-config '<your speculative config>'
```

For compatibility with the reference implementation, `VLLM_DCUT_CONFIG` and
`VLLM_ASCEND_DCUT_CONFIG` are also accepted as aliases for `DCUT_CONFIG`.
`DCUT_COST_TABLE_OUT`, `VLLM_ASCEND_DCUT_COST_TABLE_OUT`, and
`VLLM_DCUT_COST_TABLE_OUT` can override the cost-table output path without
changing the JSON file.

## Example startup exports

```bash
export VLLM_PLUGINS=ascend,dcut_adaptive_verify
export DCUT_ENABLE=1
export DCUT_CONFIG=/path/to/vllm-ascend/dcut/verify_adaptive_config.example.json
```

## Confirm D-Cut is actually cutting

Check the server log for these lines. The plugin first installs a lazy import hook,
then patches `vllm_ascend` modules after vLLM imports them normally:

```text
D-Cut lazy monkeypatch hook installed
D-Cut patched module: vllm_ascend.spec_decode.llm_base_proposer
D-Cut patched module: vllm_ascend.worker.model_runner_v1
D-Cut patched module: vllm_ascend.worker.worker
D-Cut adaptive verifier enabled
D-Cut: cost table ready
D-Cut: processing draft probabilities
D-Cut: planned adaptive draft lengths
D-Cut: cut scheduled speculative tokens
```

The last line is the direct evidence that scheduled speculative tokens were
truncated. It includes `tokens_before`, `tokens_after`, and `delta`.

For DFlash, if you see `D-Cut adaptive verifier enabled` and `D-Cut: cost table
ready` but never see `D-Cut: processing draft probabilities`, then D-Cut has not
received drafter probabilities yet. In that case enable reduce sampling so the
plugin can capture real selected-token probabilities:

```bash
# pass through vLLM additional config
--additional-config '{"enable_reduce_sample": true}'
```

For diagnosis only, `DCUT_FALLBACK_PROB=0.5` can force cost-only fallback planning
when real probabilities are unavailable. Real D-Cut decisions should use
`enable_reduce_sample=true`.
