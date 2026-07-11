# D-Cut verifier cost table

> Sample format only. Real numbers are generated during vLLM warmup and written to
> `VLLM_DCUT_COST_TABLE_MD_OUT`.

- runtime target: Qwen3.5 GDN with PIECEWISE graph capture
- num_spec_tokens: 15
- max_batch_size: 64
- warmup_seq_lens: 4096
- n_warmup_iters: 3
- n_measure_iters: 5

| batch_size | query_len_per_req | sum_query_len | cost_ms | cost_s |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | `<measured>` | `<measured>` |
| 1 | 4 | 4 | `<measured>` | `<measured>` |
| 2 | 2 | 4 | `<measured>` | `<measured>` |
| 4 | 4 | 16 | `<measured>` | `<measured>` |
| 8 | 8 | 64 | `<measured>` | `<measured>` |
| 16 | 16 | 256 | `<measured>` | `<measured>` |
| 32 | 16 | 512 | `<measured>` | `<measured>` |
| 64 | 16 | 1024 | `<measured>` | `<measured>` |
