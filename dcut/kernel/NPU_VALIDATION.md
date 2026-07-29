# D-Cut GDN NPU 验证

`dcut/tests/validate_gdn_ops_npu.py` 在目标 Ascend 设备上验证两件事：

1. 原版 `CausalConv1d` / `RecurrentGatedDeltaRule` 与对应 D-Cut 算子的
   eager 输出及原地更新 state 精度一致。
2. 两个 D-Cut 算子能够分别被 `torch.npu.NPUGraph` 捕获；改变静态
   tensor 内容后连续 replay，结果仍与同输入的 D-Cut eager 结果一致。

第二组用例包含“上一轮 accepted 位置大于本轮 query 长度”的输入，覆盖
D-Cut 固定 state 行和独立 state offset 的核心场景。

## 前置条件

先按 [README.md](README.md) 构建并安装同时包含两个算子的 AscendC 包，
然后构建 Torch 注册库。以下示例假定服务所用 CANN 是 9.0.1：

```bash
REPO_ROOT=/data/c00954457/codex_vllm/vllm-ascend
DCUT_OPP_ROOT=/usr/local/Ascend/cann-9.0.1/opp/vendors/dcut_transformer

cd "${REPO_ROOT}"
export ASCEND_CUSTOM_OPP_PATH="${DCUT_OPP_ROOT}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${DCUT_OPP_ROOT}/op_api/lib:${LD_LIBRARY_PATH:-}"
export VLLM_DCUT_TORCH_OP_LIBRARY="$PWD/dcut/kernel/build/torch_extension/dcut_torch_ops.so"

test -f "${VLLM_DCUT_TORCH_OP_LIBRARY}"
```

构建 SOC 必须与运行时识别的 SOC 一致。例如 910B 使用
`--soc=ascend910b`，A3/`ascend910_93` 使用 `--soc=ascend910_93`。

## 执行

```bash
python dcut/tests/validate_gdn_ops_npu.py \
  --device npu:0 \
  --replays 3 \
  --json-out /tmp/dcut_gdn_ops_validation.json
```

成功时每个 output/state 检查都会打印 `PASS`，最后打印：

```text
DCUT_NPU_VALIDATION_PASS metrics=...
```

任何精度不一致、算子未加载或 graph capture/replay 失败都会令脚本非零
退出。JSON 文件记录每项检查的 `max_abs`、`max_rel` 和容差。

也可以分开排查 eager 精度或图捕获：

```bash
python dcut/tests/validate_gdn_ops_npu.py --mode eager
python dcut/tests/validate_gdn_ops_npu.py --mode graph --replays 3
```

## PIECEWISE 端到端边界

生产 PIECEWISE 入图验证必须显式启用下列开关；默认值 `0` 会保留 GDN
splitting boundary，使 GDN 不入图：

```bash
export VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE=1
```

这个脚本证明算子本身的精度和 ACL Graph replay 安全性。它不能单独证明
vLLM 的实际 PIECEWISE FX 分片一定包含 GDN。完成算子验证后仍需用
PIECEWISE 启动服务并发起真实请求，同时确认启动日志包含：

```text
D-Cut: GDN core is graph-capturable in PIECEWISE mode
```

并确认 `compilation_config.splitting_ops` 不包含
`vllm::qwen_gdn_attention_core`，请求过程发生 ACL Graph replay 且无
eager fallback 或 D-Cut 算子加载失败日志。
