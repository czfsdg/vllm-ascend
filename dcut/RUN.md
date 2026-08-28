# D-Cut v0.23.0 PIECEWISE 运行手册

以下命令在 vLLM Ascend 官方容器内执行，仓库路径假定为
`/vllm-workspace/vllm-ascend`。D-Cut 只在 DFlash 或启用了
`parallel_drafting` 的 PARD 上生效。

## 1. 构建算子并安装插件

```bash
cd /vllm-workspace/vllm-ascend
bash dcut/kernel/build.sh --soc=ascend910b
python -m pip install -e ./dcut
```

`build.sh` 会把 AscendC OPP 安装到 `dcut/kernel/build/custom_ops`，插件导入时会
自动把其中的 `vendors/dcut` 加入 `ASCEND_CUSTOM_OPP_PATH`。可用
`--soc=ascend910_93` 或 `--soc=ascend950` 选择其他已支持的 SoC。

## 2. 启动 PIECEWISE 服务

```bash
export VLLM_PLUGINS=dcut_adaptive_verify
export VLLM_DCUT_CONFIG=/vllm-workspace/vllm-ascend/dcut/verify_adaptive_config.json
export VLLM_DCUT_GDN_PIECEWISE=1

cd /workspace
vllm serve /models/<target-model> \
  --served-model-name dcut-target \
  --tensor-parallel-size <tp-size> \
  --max-model-len <max-model-len> \
  --max-num-seqs 16 \
  --speculative-config \
    '{"method":"dflash","model":"/models/<dflash-model>","num_speculative_tokens":15}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --port 8000
```

开关默认关闭，并且只有同时设置 `VLLM_DCUT_CONFIG` 时才会展开 GDN recurrent
core。旧启动脚本中的 `VLLM_ASCEND_ENABLE_DCUT_GDN_PIECEWISE=1` 仍作为兼容
别名支持。图内 GDN 当前要求 PCP=1、DCP=1；其他并行配置保留 PIECEWISE
外层图，但整个 GDN 回退为 eager boundary。Ascend 310P 不启用该 GDN 路径。

## 3. 验证

```bash
curl -sf http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dcut-target","messages":[{"role":"user","content":"say hi"}],"temperature":0,"max_tokens":16}'
```

PIECEWISE GDN 成功捕获时应出现：

```text
D-Cut: captured PIECEWISE GDN graph with recurrent update inside
```

启动成功但首个请求失败不算通过。最终验证必须使用真实权重，并确认响应为 HTTP
200 且 `choices` 非空；`--load-format dummy` 只能用于先排查架构和算子路径。
