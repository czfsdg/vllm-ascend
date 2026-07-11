# NPU Qwen3.5-9B D-Cut startup

这个脚本按当前 NPU 环境写死了默认值，不需要改任何一行：

```bash
bash dcut/start_npu_qwen35_9b_dcut.sh
```

脚本默认导出：

```bash
ASCEND_RT_VISIBLE_DEVICES=11
VLLM_DCUT_CONFIG=/data/c00954457/codex_vllm/vllm-ascend/dcut/verify_adaptive_config.example.json
VLLM_DCUT_COST_TABLE_OUT=/data/c00954457/codex_vllm/vllm-ascend/dcut/cost_table.json
VLLM_DCUT_TRIM_STATS_OUT=/data/c00954457/codex_vllm/vllm-ascend/dcut/trim_stats.txt
VLLM_DCUT_STAT_EVERY=1
VLLM_DCUT_PROFILE_FORCE_EAGER=1
VLLM_USE_V1=1
VLLM_ASCEND_MODEL_PLUGIN=vllm_ascend.patch_qwen3_5
VLLM_PLUGINS=ascend,dcut_adaptive_verify
VLLM_TARGET_DEVICE=ascend
```

启动命令使用 `python3 -m vllm.entrypoints.openai.api_server`，模型为 `/data/models/Qwen3.5-9B`，draft 模型为 `/data/models/Qwen3.5-9B-DFlash`，端口 `8305`，TP=1，并把 served model names 设置为 `qwen35 qwen3.5-9b`，因此请求里的 `model=qwen3.5-9b` 也能命中；并使用：

```bash
-cc '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512]}'
--speculative-config '{"method":"dflash","model":"/data/models/Qwen3.5-9B-DFlash","num_speculative_tokens":7,"enforce_eager":true}'
```

## Cost table 没生成时先看这里

`VLLM_PLUGINS` 必须同时包含 `ascend` 和 `dcut_adaptive_verify`；如果只写 `ascend`，D-Cut 的 monkey patch 可能不会加载，`VLLM_DCUT_CONFIG` / `VLLM_DCUT_COST_TABLE_OUT` 也就不会被消费。当前脚本默认设置为：

```bash
VLLM_PLUGINS=ascend,dcut_adaptive_verify
```

## 结果文件

- `/data/c00954457/codex_vllm/vllm-ascend/dcut/cost_table.json`：warmup/profile 后的 verifier cost table。
- `/data/c00954457/codex_vllm/vllm-ascend/dcut/trim_stats.txt`：D-Cut 每步裁剪统计，用于判断是否真的发生了裁剪。

`trim_stats.txt` 每行格式类似：

```text
step=1 batch_size=4 scheduled_reqs=4 total_scheduled_tokens=32 trimmed_reqs=2 trimmed_tokens=5 total_trimmed_reqs=2 total_trimmed_tokens=5
```

其中 `trimmed_tokens > 0` 表示这一 step D-Cut 实际切掉了 verifier 要检查的 speculative tokens。

## 服务启动日志里的 D-Cut debug 信息

启动脚本会在执行 `api_server` 前打印这些行，先确认路径和插件是否正确：

```text
[dcut-start] VLLM_PLUGINS=ascend,dcut_adaptive_verify
[dcut-start] VLLM_DCUT_CONFIG=.../verify_adaptive_config.example.json
[dcut-start] VLLM_DCUT_COST_TABLE_OUT=.../cost_table.json
[dcut-start] VLLM_DCUT_TRIM_STATS_OUT=.../trim_stats.txt
```

服务日志中应继续看到这些关键字：

```text
D-Cut install requested
D-Cut adaptive-verify monkey patch installed
D-Cut patched execute_model for vllm.v1.worker.gpu_model_runner.GPUModelRunner
D-Cut runner init concrete class: ...
D-Cut patched execute_model for vllm_ascend.worker.model_runner_v1.NPUModelRunner
D-Cut Ascend worker module watch installed; waiting for Worker class definition.
D-Cut patched worker hooks for vllm_ascend.worker.worker.Worker
D-Cut adaptive verify ENABLED
D-Cut worker warmup hook reached: vllm_ascend.worker.worker.Worker.compile_or_warm_up_model
D-Cut cost profiling START
# 如果 warmup hook 没走，第一次请求时应看到：
D-Cut cost profiling LAZY START from vllm_ascend.worker.worker.Worker.execute_model
VerifyAdaptiveController: begin cost profiling
VerifyAdaptiveController: profile row ... avg_ms=...
VerifyAdaptiveController: dumped JSON cost table to .../cost_table.json
D-Cut cost profiling END
```

如果没有 `D-Cut install requested`，说明 `dcut_adaptive_verify` 插件没有加载；如果看到 `Ascend NPUModelRunner patch skipped` 或循环导入错误，说明还在用旧脚本/旧代码。当前代码不会在 plugin install 阶段主动 import Ascend runner，而是在 runner 实例初始化完成后检查已经加载的 `vllm_ascend.worker.worker.Worker` 并动态 patch，因此应在 `D-Cut adaptive verify ENABLED` 附近看到 `D-Cut runner init concrete class`、`D-Cut patched execute_model for vllm_ascend.worker.model_runner_v1.NPUModelRunner`，以及 `D-Cut Ascend worker module watch installed` 或 `D-Cut patched worker hooks for vllm_ascend.worker.worker.Worker`。当前 vLLM 启动路径如果仍然不走 worker warmup hook，第一次请求应触发 `D-Cut cost profiling LAZY START from vllm_ascend.worker.worker.Worker.execute_model` 并生成 cost table。
