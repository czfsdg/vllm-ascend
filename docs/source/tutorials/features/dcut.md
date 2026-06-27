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

This installs the `vllm-ascend-dcut` package and exposes a `dcut` vLLM
general-plugin entrypoint. It does **not** reinstall `vllm-ascend`.

## Enable in the startup script

D-Cut is disabled by default. Add the enable flag before starting vLLM:

```bash
export VLLM_ASCEND_ENABLE_DCUT=1
vllm serve <model> --speculative-config '<your speculative config>'
```

If your environment restricts vLLM plugins through `VLLM_PLUGINS`, append `dcut`
to the existing list instead of replacing the Ascend plugin entries, for example:

```bash
export VLLM_PLUGINS="${VLLM_PLUGINS},dcut"
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

export VLLM_ASCEND_ENABLE_DCUT=1
export VLLM_ASCEND_DCUT_CONFIG=/tmp/dcut_config.json
vllm serve <model> --speculative-config '<your speculative config>'
```

`VLLM_ASCEND_DCUT_COST_TABLE_OUT` can also be used to override the cost-table
output path without changing the JSON file.
