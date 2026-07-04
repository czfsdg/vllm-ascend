# dcut adaptive verifier

Standalone D-Cut adaptive verifier step-length plugin for vLLM 0.22.x.

## Install

```bash
cd dcut
pip install -e .
```

## Enable

```bash
export VLLM_DCUT_CONFIG=/path/to/verify_adaptive_config.json
vllm serve <model> --speculative-config '{"method":"dflash","model":"<draft>","num_speculative_tokens":15}'
```

The plugin prints every target verification result after verification finishes:

- total draft tokens before truncation
- total draft tokens verified after D-Cut truncation
- per-request `original_len`, `verify_len`, and `cut_tokens`
- target verifier elapsed time in milliseconds

See `verify_adaptive_config.example.json` for the config schema.
