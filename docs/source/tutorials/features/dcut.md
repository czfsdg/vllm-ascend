# D-Cut adaptive verifier

D-Cut is an optional plugin-style feature for Ascend speculative decoding. It
adapts verifier draft length per request by profiling verifier cost and using
the selected draft-token probabilities from the drafter.

## Install

From the vLLM Ascend repository:

```bash
pip install -e .
```

## Enable

Enable D-Cut explicitly after installing the editable package:

```bash
export VLLM_ASCEND_ENABLE_DCUT=1
vllm serve <model> --speculative-config '<your speculative config>'
```

D-Cut is disabled by default. Installing the package alone does not change
runtime behavior. The editable install exposes a `dcut` vLLM general-plugin
entrypoint. In normal vLLM plugin discovery no extra `VLLM_PLUGINS` setting is
needed. If your environment already restricts vLLM plugins with `VLLM_PLUGINS`,
append `dcut` to the existing Ascend plugin list rather than replacing it.

## Optional config

The built-in defaults are used when `VLLM_ASCEND_ENABLE_DCUT=1` and no config is
provided. To override profiling levels or dump the cost table:

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
