# D-Cut GDN 自定义算子

这里包含两个仅服务于 D-Cut 变长 speculative decode 的 AscendC 算子，所有新增和修改的源码都在 `dcut/` 下：

- `dcut_recurrent_gated_delta_rule`：输入固定槽位矩阵 `ssm_state_indices[B, S]` 和上一轮的 `num_accepted_tokens[B]`。它按 request 行读取上一轮接受位置的 recurrent state，再只写回本轮有效 token 对应的槽位。
- `dcut_causal_conv1d`：通过 `state_offsets[B]` 读取每个 request 上一轮的卷积窗口；`query_start_loc[B+1]` 独立描述本轮的变长 token。

这样，上一轮接受位置可以大于本轮 D-Cut 长度，不再需要 `target = max(random_len, gdn_min)`。索引都保留为 device tensor，Python 热路径没有 `item()`。

## 目录

- `dcut/kernel/dcut_recurrent_gated_delta_rule/`：recurrent 算子的 op host、op kernel 和 ACLNN 适配。
- `dcut/kernel/dcut_causal_conv1d/`：conv1d 算子的 op host、op kernel 和 ACLNN 适配。
- 每个算子的 `vendor/`：从 vllm-ascend 0.23.0 基线复制并局部修改的实现，避免改动 `csrc/`。
- `dcut/kernel/torch_extension/`：两个 `torch.ops._C_ascend` schema、PrivateUse1 实现和 Meta 实现。

## 单独编译

在已经安装 CANN、torch 和 torch_npu 的 Linux/NPU 环境中，从仓库根目录执行：

```bash
# 两个 AscendC 算子和 Torch 注册库一起构建
bash dcut/kernel/build.sh --soc=ascend910b

# 也可以只构建一个 AscendC 算子
bash dcut/kernel/build.sh \
  --ops=dcut_recurrent_gated_delta_rule \
  --soc=ascend910b
bash dcut/kernel/build.sh \
  --ops=dcut_causal_conv1d \
  --soc=ascend910b
```

A3 使用 `--soc=ascend910_93`，Ascend 950 使用 `--soc=ascend950`；310P 暂未接入。脚本通过临时符号链接复用 vllm-ascend 的原生 custom-op 工具链，并使用独立的 `dcut` vendor，退出时会删除链接，不会修改或覆盖 `csrc/` 源码和已有的 `custom_transformer` 包。AscendC 安装包输出到 `csrc/output/*.run`。

`--ops` 适合分别编译和排查单个算子；实际运行 D-Cut 前，建议使用默认命令把两个算子构建到同一个 `dcut_transformer` 安装包。

Torch 注册库可单独构建：

```bash
bash dcut/kernel/build.sh --torch-only
```

默认输出为：

```text
dcut/kernel/build/torch_extension/dcut_torch_ops.so
```

D-Cut 插件会在 worker 初始化、ACL Graph 捕获之前自动加载这个默认路径。如果库放在别处，设置：

```bash
export VLLM_DCUT_TORCH_OP_LIBRARY=/absolute/path/dcut_torch_ops.so
```

## PIECEWISE 入图边界

vLLM Ascend 0.23 的 `forward` 和 `torch.ops.vllm.qwen_gdn_attention_core` 保持不变，后者继续作为 PIECEWISE splitting op。D-Cut 只替换 splitting op 调用的 `_forward_core`，并只在 speculative 分支调用这两个新算子；prefill、non-spec 分支和 GDN metadata 生命周期不变。

打开 `VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE=1` 时，该 splitting op
内部会为 pure-spec decode 按 padded token bucket 捕获/回放局部 ACL Graph；
prefill/mixed 仍只在这个 boundary eager，外层 PIECEWISE segment 不会降级。
两个新 schema 都提供 Meta 实现，描述输出 shape 和 state alias，供 `torch.compile`/ACL Graph 做 shape 与副作用分析。最终仍需要在目标 NPU 上跑 PIECEWISE ACL Graph 的精度和 replay 验证。
