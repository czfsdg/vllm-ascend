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
  "min_warmup_batch_size": 1,
  "batch_size_step": 1,
  "budget_ratios": [0.25, 0.5, 0.75, 1.0],
  "max_warmup_batch_size": 64,
  "query_len_step_per_req": 2,
  "min_query_len_per_req": 2,
  "n_warmup_iters": 3,
  "n_measure_iters": 5,
  "log_decision_details": true,
  "log_decision_interval": 1,
  "log_decision_max_records": 8,
  "log_verifier_timing": true,
  "log_verifier_timing_interval": 1,
  "log_attention_query_shape": true,
  "log_attention_query_shape_interval": 1,
  "min_score_improvement_ratio": 0.0,
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

To inspect why D-Cut often keeps full draft lengths, set
`log_decision_details=true` in the JSON config. This prints the per-position mean
selected probabilities and every measured candidate's `(Q, S, cost_ms, score)`.
Use `log_decision_interval` to print every Nth decision and
`log_decision_max_records` to cap the number of candidate rows in each log.

Set `log_verifier_timing=true` to print synchronized per-verifier-step timing
logs with the post-cut scheduled token count and speculative token count. This
adds an NPU synchronization around each logged step, so use
`log_verifier_timing_interval` to reduce overhead during longer benchmark runs.

Set `log_attention_query_shape=true` to print the query lengths passed into
attention metadata (`num_tokens`, `num_tokens_padded`, `max_query_len`, and
`query_lens`). Use this to verify whether the attention path still computes a
fixed or padded query shape after D-Cut truncates speculative tokens.

`batch_size_step=1` profiles every batch size by default, so a runtime batch
size of 3 uses budget buckets built for batch size 3 instead of being rounded up
to the next profiled batch size.

D-Cut uses `budget_ratios` to build batch-level verify budget candidates. For
example, with `num_speculative_tokens=7`, `batch_size=16`, and
`budget_ratios=[0.25, 0.5, 0.75, 1.0]`, the speculative budgets are
`ceil(ratio * 16 * 7)` tokens across the whole batch.

By default, `min_score_improvement_ratio=0.0` makes the planner choose the
highest-scoring budget directly. Increase it only for diagnostics when you
intentionally want to bias toward shorter budgets and require a longer budget to
win by a minimum relative margin.
