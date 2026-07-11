# NPU Qwen3.5-9B D-Cut startup

这个脚本按当前 NPU 环境写死了默认值，不需要改任何一行：

```bash
bash dcut/start_npu_qwen35_9b_dcut.sh
```

脚本默认导出：

```bash
ASCEND_RT_VISIBLE_DEVICES=11
VLLM_DCUT_CONFIG=/data/c00954457/codex_vllm/vllm-ascend/dcut/verify_adaptive_config.example.json
VLLM_DCUT_COST_TABLE_OUT=/data/c00954457/cost_table.json
VLLM_DCUT_TRIM_STATS_OUT=/data/c00954457/trim_stats.txt
VLLM_DCUT_STAT_EVERY=1
VLLM_DCUT_PROFILE_FORCE_EAGER=0
VLLM_USE_V1=1
VLLM_ASCEND_MODEL_PLUGIN=vllm_ascend.patch_qwen3_5
VLLM_PLUGINS=dcut_adaptive_verify
```

启动命令使用 `python3 -m vllm.entrypoints.openai.api_server`，模型为 `/data/models/Qwen3.5-9B`，draft 模型为 `/data/models/Qwen3.5-9B-DFlash`，端口 `8305`，TP=1，并使用：

```bash
-cc '{"cudagraph_mode":"piecewise","cudagraph_capture_sizes":[1,2,4,8,16,32,64,128,256,512]}'
--speculative-config '{"method":"dflash","model":"/data/models/Qwen3.5-9B-DFlash","num_speculative_tokens":7,"enforce_eager":true}'
```

## 结果文件

- `/data/c00954457/cost_table.json`：warmup/profile 后的 verifier cost table。
- `/data/c00954457/trim_stats.txt`：D-Cut 每步裁剪统计，用于判断是否真的发生了裁剪。

`trim_stats.txt` 每行格式类似：

```text
step=1 batch_size=4 scheduled_reqs=4 total_scheduled_tokens=32 trimmed_reqs=2 trimmed_tokens=5 total_trimmed_reqs=2 total_trimmed_tokens=5
```

其中 `trimmed_tokens > 0` 表示这一 step D-Cut 实际切掉了 verifier 要检查的 speculative tokens。
