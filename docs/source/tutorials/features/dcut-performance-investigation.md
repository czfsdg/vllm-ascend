# D-Cut 自适应 verifier 性能探索记录

本文档汇总截至目前对 D-Cut 自适应 verifier 的性能探索、关键日志、阶段性结论和下一步建议。目标是先把已经验证过的现象沉淀下来，避免后续继续定位时重复走弯路。

## 背景

D-Cut 的目标是在 vLLM Ascend speculative decoding 场景中，根据 drafter 选中 token 的概率和 verifier cost profile，自适应缩短每个请求的 speculative draft length。理想流程如下：

1. 从 drafter 侧收集每个 speculative token 的概率；
2. 对不同 verifier token budget 做 cost profiling；
3. 在每个 decode batch 上选择一组 per-request draft length；
4. 在 verifier 执行前裁剪 scheduled speculative tokens；
5. 当 verifier query tokens 变少时，降低 verifier latency。

最初假设是：如果 D-Cut 能把 verifier query tokens `Q` 降下来，verifier cost 应该会下降，从而抵消 draft length 变短带来的 acceptance 损失。但当前探索显示，D-Cut 功能上确实生效了，verifier input 也确实变小了，不过 verifier 端到端耗时没有随 `Q` 明显下降。

## 已完成探索时间线

### 1. 确认 D-Cut 是否真的裁剪了 scheduled speculative tokens

早期日志里可以看到 planner 产出了非全 7 的 draft length，例如 `[7, 2, 5, 7]`、`[7, 3, 5]`、`[7, 4, 4]`。scheduler 侧也能看到实际裁剪日志：

```text
D-Cut: cut scheduled speculative tokens reqs=3 tokens_before=24 tokens_after=18 delta=6
D-Cut verifier input check: ... total_num_scheduled_tokens=18
```

这一步排除了“D-Cut 只是计划了较短长度，但没有作用到 verifier input”的可能性。

**阶段结论**：D-Cut 不只是产生了较短 draft length，实际 verifier input metadata 也被改小了。

### 2. 修正 cost profiling，只测 verifier target cost

初始 profiling 路径可能把 drafter 的 dummy/warmup 成本也算进 verifier cost，导致 cost table 不纯。后续给 `_dummy_run` 增加了 `skip_drafter` profiling 路径，让 D-Cut 测 verifier cost 时可以跳过 drafter。

过程中出现过兼容性错误：

```text
NPUModelRunner._dummy_run() got an unexpected keyword argument 'skip_drafter'
```

这个错误说明运行时 runner 还不支持新参数，或者实际加载的是旧版本实现。后续通过 Ascend model runner 集成 `skip_drafter` 参数，并在 profiling 调用侧兼容旧签名，解决了这个问题。

**阶段结论**：profiling 应该只测 verifier target cost，不应该把 drafter 成本混进去；但修正后 cost curve 仍然接近水平。

### 3. 直接统计每次 verifier wall time

随后增加了 verifier 实际执行耗时日志，典型日志如下：

```text
D-Cut verifier timing: elapsed_ms=70.592 total_tokens=25 spec_reqs=4 spec_tokens=21 avg_spec_len=5.25
D-Cut verifier timing: elapsed_ms=69.211 total_tokens=32 spec_reqs=4 spec_tokens=28 avg_spec_len=7.00
```

这里最关键的现象是：`total_tokens` 从 `32` 降到 `25` 后，verifier wall time 并没有明显下降，甚至很多 batch 基本一样。后续更大 batch 的日志也复现了同样现象。

**阶段结论**：scheduled token count 变小了，但 verifier wall time 基本不变。

### 4. 检查 Ascend attention query shape 是否真的变小

因为 verifier time 没下降，下一步怀疑 Ascend graph、compiled shape 或 bucket 机制可能仍然按最大 shape 跑。因此增加了 attention query shape 日志，典型输出如下：

```text
D-Cut attention query shape: num_tokens=64 ... query_lens=[8, 8, 8, 8, 8, 8, 8, 8]
D-Cut attention query shape: num_tokens=50 ... query_lens=[8, 6, 8, 8, 6, 3, 8, 3]
```

这说明 D-Cut 裁剪后，attention metadata 里看到的 `num_tokens` 确实从 `64` 下降到了 `50`。

**阶段结论**：`Q` 在 verifier attention metadata 层面确实变小了，不是所有请求还保持原始 draft length。

### 5. 统计 attention backend 时间

确认 query shape 确实变小后，又给 Ascend attention backend 增加了 timing。典型结果如下：

```text
Q=64: attention elapsed_ms=2.812, query_tokens=512
Q=50: attention elapsed_ms=2.853, query_tokens=400
```

虽然 `query_tokens` 从 `512` 降到 `400`，下降约 22%，但 attention 时间仍然在 `2.8 ms` 左右。在后续 `Q=57` 的日志中，attention 时间也基本是 `2.7-2.9 ms`。

**阶段结论**：当前 workload 和 shape 范围下，标准 attention 不是 verifier 主瓶颈；attention 更像是固定 overhead、launch overhead、bucket/graph overhead 或 backend 调度 overhead 主导，而不是随小范围 `Q` 线性变化。

### 6. 将 verifier wall time 拆成粗粒度 phase

attention 只有约 `2.8 ms`，不足以解释 `65-110 ms` 的 verifier runtime。因此继续把 verifier 拆成 `prepare_inputs`、`build_attention_metadata`、`model_forward`、`compute_logits` 等 phase。一个 full verification 样例如下：

```text
total_ms=108.800
phases={
  'build_attention_metadata': 6.415,
  'compute_logits': 1.863,
  'determine_batch_execution': 0.073,
  'model_forward': 97.199,
  'prepare_inputs': 2.070,
  'preprocess': 0.281,
  'sanitize_placeholder_input_ids': 0.024,
  'update_states': 0.216,
}
```

一个 D-Cut 后的样例如下：

```text
total_ms=113.416
total_tokens=50
phases={
  'build_attention_metadata': 7.281,
  'compute_logits': 1.823,
  'model_forward': 100.817,
  'prepare_inputs': 2.179,
}
```

phase breakdown 显示瓶颈非常集中：

| 组件 | 近似耗时 | 解读 |
| --- | ---: | --- |
| `model_forward` | 97-103 ms | 主瓶颈 |
| attention backend | 2.7-2.9 ms | verifier 总耗时的小部分 |
| attention metadata build | 6-7 ms | 有一定固定开销 |
| logits | 1.8-1.9 ms | 较小 |
| prepare inputs | 2.0-2.5 ms | 较小 |

**阶段结论**：剩余大头在 `model_forward` 内部，而且 measured attention backend 不能解释这部分耗时。

### 7. 尝试按 module 继续拆 `model_forward`

为了继续拆 `model_forward`，增加了可选的 model leaf module timing hook。但最新日志中仍然看到：

```text
module_classes=[] module_names=[]
```

这说明本次运行没有采到有效 module-level timing。可能原因包括：

1. 运行配置没有打开 `log_model_forward_module_breakdown`；
2. 修改配置后没有重启 engine；
3. 实际加载的是旧 plugin 或旧 runtime；
4. model forward 被 compile、graph 或 custom op 包装，执行路径没有经过 Python leaf module hook。

**阶段结论**：下一步首先要确认 module timing hook 是否真的安装。如果确认安装后仍然为空，就需要低于 Python module hook 的 profiler 或打点方式。

## 最新日志的核心含义

最新日志中同时出现 full 和 cut batch：

```text
Q=64 total_ms=108.800 model_forward=97.199 attention=2.812
Q=50 total_ms=113.416 model_forward=100.817 attention=2.853
```

这组数据说明三件事。

### Q 的下降是真实的

`total_tokens` 和 `query_tokens` 在 D-Cut 后都下降了。因此不能说 D-Cut 没生效，也不能说 verifier input 仍然是 full shape。

### Attention 对这个 Q 范围不敏感

attention query token 数从 `512` 降到 `400`，但 attention time 基本不变。因此，即使 D-Cut 减少了 attention query tokens，也无法带来明显端到端收益。

### Planner 目前高估了切分收益

当前 cost table 基本是平的，例如：

| Candidate Q | Cost |
| ---: | ---: |
| 22 | 34.3265 ms |
| 36 | 34.2986 ms |
| 50 | 33.4220 ms |
| 64 | 33.5338 ms |

profile table 中，`Q=50` 相比 `Q=64` 只快约 `0.3%`，live verifier timing 甚至可能更慢。这个差异太小，不足以抵消 acceptance length 下降和 runtime 噪声。

## 当前工作假设

D-Cut 当前功能是 active 的，但当前 Ascend verifier runtime 对测试到的 `Q` 范围并不敏感。dominant cost 在 `model_forward` 内部，而 measured attention backend 只占 verifier 总耗时的一小部分。

可能解释如下：

1. **固定 overhead 主导**：graph dispatch、NPU launch、backend scheduling、metadata construction 等固定开销盖过了裁剪少量 query token 的收益；
2. **bucket 或 padded kernel 成本仍然接近固定**：即使开 eager，一些内部 kernel、融合 op 或 backend 路径仍可能按 bucket/fixed shape 执行；
3. **非 attention 层主导**：MLP、norm、projection、logits 或 fused model block 可能占主要耗时，而这些部分不一定随 verifier query token 数明显下降；
4. **Python timing 只能用于诊断**：当前 timing 为了分 phase 会插入同步，绝对耗时会被放大；但 phase 相对比例仍然有价值。

## 推荐下一步

### A. 先确认 model-forward module hook 是否安装成功

使用类似配置：

```json
{
  "log_verifier_timing": true,
  "log_verifier_breakdown": true,
  "log_attention_query_shape": true,
  "log_attention_timing": true,
  "log_model_forward_module_breakdown": true,
  "log_model_forward_module_top_k": 20
}
```

重启 engine 后先找这条日志：

```text
D-Cut: installed model-forward module timing hooks count=...
```

如果没有这条日志，说明 module hook 没安装；如果有正数 `count`，但后续 `module_classes=[]` 仍为空，则说明执行路径大概率绕过了 Python module hook。

### B. 如果 Python module hook 生效，找出 top modules

如果 module hook 能采到数据，下一步看 `model_forward` 内部到底是谁消耗了 100ms，例如：

- attention wrapper 和非 attention 层的比例；
- MLP/feed-forward；
- projection layer；
- normalization；
- logits/output projection；
- custom fused block。

这个结果能决定后续是否还有 D-Cut 方向的优化空间。

### C. 如果 Python module hook 不生效，转向更低层 profiler

如果确认 hook 安装成功但 module 统计仍为空，建议转向：

1. 在 Ascend custom op 或 backend wrapper 里打点；
2. 在 model executor 更靠近 compiled call 的函数边界打点；
3. 对 verifier step 跑 Ascend/CANN profiler trace；
4. 临时关闭 compile/fusion，只用于诊断，对比 optimized path 和 eager path 的模块级耗时。

### D. 给 planner 加 cost-gain guard

由于当前 cost table 基本是平的，planner 不应该在 cost reduction 很小的时候继续切。建议增加一个保护条件：

```text
if (full_cost - selected_cost) / full_cost < min_cost_gain_ratio:
    use full draft length
```

初始阈值可以设为 `3%` 或 `5%`。按照当前最新 cost table，`Q=50` 相比 `Q=64` 只有约 `0.3%` 的 profile cost 改善，因此这个 guard 会选择 full verification，避免无收益切分。

## 阶段性结论

基于目前证据，D-Cut 还不能算这个 workload 上的性能收益点。它确实减少了 verifier input size，但 runtime cost 没有随之下降。继续调 planner 权重意义不大，下一步更有价值的是解释为什么 `model_forward` 在 full 和 cut batch 下都维持在约 `100 ms`。

