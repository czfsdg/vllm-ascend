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

### Install and validate the AscendC package

The default build produces one package containing both D-Cut operators. Install
that package into the same CANN OPP tree used by the service:

```bash
DCUT_RUN="$(ls -t csrc/output/*.run | head -n 1)"
test -n "${DCUT_RUN}"
"${DCUT_RUN}" \
  --install-path=/usr/local/Ascend/cann-9.0.1/opp
```

Before starting vLLM, verify that the installed custom opapi library exports
both API phases for both operators:

```bash
DCUT_OPP_ROOT=/usr/local/Ascend/cann-9.0.1/opp/vendors/dcut_transformer
DCUT_CUST_OPAPI="${DCUT_OPP_ROOT}/op_api/lib/libcust_opapi.so"

test -f "${DCUT_CUST_OPAPI}"
if ldd "${DCUT_CUST_OPAPI}" | grep -q 'not found'; then
  ldd "${DCUT_CUST_OPAPI}"
  exit 1
fi
nm -D "${DCUT_CUST_OPAPI}" | grep -E \
  'aclnnDcut(CausalConv1d|RecurrentGatedDeltaRule)(GetWorkspaceSize)?$'
```

The final command must list four symbols. Point the existing ACLNN loader at
this vendor before the Torch registration library is loaded:

```bash
export ASCEND_CUSTOM_OPP_PATH="${DCUT_OPP_ROOT}:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${DCUT_OPP_ROOT}/op_api/lib:${LD_LIBRARY_PATH:-}"
```

`ASCEND_CUSTOM_OPP_PATH` is the manual custom-op loading control used by the
adapter: it opens each vendor's `op_api/lib/libcust_opapi.so` and resolves the
requested ACLNN symbol there. No extra D-Cut environment variable or
`LD_PRELOAD` is required.

Torch 注册库可单独构建：

```bash
bash dcut/kernel/build.sh --torch-only -j8
```

默认输出为：

```text
dcut/kernel/build/torch_extension/dcut_torch_ops.so
```

`-j8` 显式指定 Torch 注册库的并行编译任务数，可按机器 CPU
资源调整。在容器采用双源码目录时，例如主 `vllm-ascend` editable
安装位于 `/vllm-workspace/vllm-ascend`、D-Cut 源码位于 `/data` 下的
独立 clone，应从 D-Cut clone 的仓库根目录执行上述构建命令；不需要
修改或重新安装 `/vllm-workspace/vllm-ascend`。

D-Cut 插件会在 worker 初始化、ACL Graph 捕获之前自动加载这个默认路径。如果库放在别处，设置：

```bash
export VLLM_PLUGINS=ascend,dcut_adaptive_verify
export VLLM_DCUT_TORCH_OP_LIBRARY=\
/absolute/path/to/vllm-ascend/dcut/kernel/build/torch_extension/dcut_torch_ops.so

if [[ ! -f "${VLLM_DCUT_TORCH_OP_LIBRARY}" ]]; then
  echo "Missing D-Cut Torch ops: ${VLLM_DCUT_TORCH_OP_LIBRARY}"
  exit 1
fi
```

不需要设置 `LD_PRELOAD`，也不需要在启动前用单独的 Python 进程保持
算子库加载状态。插件会在每个需要的进程中调用
`torch.ops.load_library()`。

启动服务前可用下面的命令验证 schema、NPU 实现和 Meta 实现均已注册：

```bash
python - <<'PY'
import os

import torch

library_path = os.environ["VLLM_DCUT_TORCH_OP_LIBRARY"]
torch.ops.load_library(library_path)

op_names = (
    "npu_dcut_causal_conv1d",
    "npu_dcut_recurrent_gated_delta_rule",
)
for op_name in op_names:
    qualified_name = f"_C_ascend::{op_name}"
    print("operator:", getattr(torch.ops._C_ascend, op_name))
    print(
        "schema:",
        torch._C._dispatch_find_schema_or_throw(qualified_name, "").schema(),
    )
    print(
        "PrivateUse1:",
        torch._C._dispatch_has_kernel_for_dispatch_key(
            qualified_name, "PrivateUse1"
        ),
    )
    print(
        "Meta:",
        torch._C._dispatch_has_kernel_for_dispatch_key(
            qualified_name, "Meta"
        ),
    )
PY
```

两个算子的 `PrivateUse1` 和 `Meta` 都应输出 `True`。算子名包含
`npu_` 前缀；不带前缀的 `dcut_causal_conv1d` 和
`dcut_recurrent_gated_delta_rule` 不是注册名。

## PIECEWISE 入图边界

vLLM Ascend 0.23 的 `forward` 和 `torch.ops.vllm.qwen_gdn_attention_core` 保持不变。D-Cut 在插件加载早期仅从 PIECEWISE `splitting_ops` 中移除 `vllm::qwen_gdn_attention_core`，因此该 custom op 及其调用的两个新算子会随所在 FX 分片一起被 ACL Graph 捕获；其余 attention splitting 边界不变。D-Cut 只替换 custom op 调用的 `_forward_core`，并只在 speculative 分支调用这两个新算子；prefill、non-spec 分支和 vLLM 0.23 `GDNSpecDecodeMetadata` 生命周期不变。

启动日志应包含：

```text
D-Cut: GDN core is graph-capturable in PIECEWISE mode
```

同时打印出的 `compilation_config.splitting_ops` 中不应再包含
`vllm::qwen_gdn_attention_core`。这两个条件用于确认 GDN 不再作为
PIECEWISE 图间 eager 边界执行。

两个新 schema 都提供 Meta 实现，描述输出 shape 和 state alias，供 `torch.compile`/ACL Graph 做 shape 与副作用分析。最终仍需要在目标 NPU 上跑 PIECEWISE ACL Graph 的精度和 replay 验证。
