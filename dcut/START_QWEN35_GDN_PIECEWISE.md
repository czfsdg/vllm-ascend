# Qwen3.5 GDN PIECEWISE graph + D-Cut startup example

目标：Qwen3.5 的 GDN 路径走 `PIECEWISE` 入图，同时让 D-Cut 在启动 warmup 后直接输出 verifier cost table。

## 1. 安装 dcut 插件

```bash
cd /workspace/vllm-ascend
python3 -m pip install -e dcut
```

## 2. 准备 D-Cut cost table 配置

```bash
cp dcut/qwen35_gdn_piecewise_config.example.json /tmp/qwen35_gdn_piecewise_dcut.json
```

默认配置会覆盖 batch size `1,2,4,8,16,32,64` 和 query length `2,4,...,16`，warmup seq len 是 `4096`，启动后会写出：

- `/tmp/qwen35_gdn_piecewise_cost_table.json`
- `/tmp/qwen35_gdn_piecewise_cost_table.md`

也会在 server log 里打印 Markdown 格式的 cost table，便于你直接确认 `runtime_mode` / `cost_ms`。

## 3. 启动 Qwen3.5 GDN PIECEWISE server

根据你的模型路径替换 `MODEL`、`DRAFT_MODEL` 和 TP 数。重点是：

- `--compilation-config '{"cudagraph_mode":"PIECEWISE"}'`：强制 PIECEWISE 入图。
- `VLLM_DCUT_COST_TABLE_OUT`：JSON cost table。
- `VLLM_DCUT_COST_TABLE_MD_OUT`：Markdown cost table。
- `--no-async-scheduling`：D-Cut 当前建议关闭 async scheduling。

```bash
export MODEL=/data/models/Qwen3.5-GDN
export DRAFT_MODEL=/data/models/Qwen3.5-GDN-Draft
export DCUT_CONFIG=/tmp/qwen35_gdn_piecewise_dcut.json
export VLLM_DCUT_CONFIG=${DCUT_CONFIG}
export VLLM_DCUT_COST_TABLE_OUT=/tmp/qwen35_gdn_piecewise_cost_table.json
export VLLM_DCUT_COST_TABLE_MD_OUT=/tmp/qwen35_gdn_piecewise_cost_table.md
export VLLM_PLUGINS=dcut_adaptive_verify
export VLLM_LOGGING_LEVEL=INFO

vllm serve ${MODEL} \
  --served-model-name qwen35-gdn \
  --host 0.0.0.0 \
  --port 8100 \
  --tensor-parallel-size 8 \
  --max-num-seqs 64 \
  --max-model-len 4096 \
  --speculative-config '{"method":"dflash","model":"'"${DRAFT_MODEL}"'","num_speculative_tokens":15}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --no-async-scheduling
```

## 4. 查看 cost table

```bash
cat /tmp/qwen35_gdn_piecewise_cost_table.md
python3 -m json.tool /tmp/qwen35_gdn_piecewise_cost_table.json | head -80
```

日志里应能看到类似：

```text
VerifyAdaptiveController: cost table ready (... entries).
D-Cut verifier cost table (Qwen3.5 GDN PIECEWISE):
| batch_size | query_len_per_req | sum_query_len | cost_ms | cost_s |
...
```

如果没有看到 cost table，优先检查：

1. `VLLM_PLUGINS=dcut_adaptive_verify` 是否生效。
2. `VLLM_DCUT_CONFIG` 是否指向可读 JSON。
3. speculative config 是否是 `dflash` 或 PARD parallel drafting。
4. server 是否真的完成了 `compile_or_warm_up_model`。

## NPU one-shot script

按照当前 NPU 环境，直接运行下面这个脚本即可，不需要手改命令行：

```bash
bash dcut/start_npu_qwen35_9b_dcut.sh
```

兼容入口 `bash dcut/start_qwen35_gdn_piecewise.sh` 也会转到同一个 NPU 启动脚本。
