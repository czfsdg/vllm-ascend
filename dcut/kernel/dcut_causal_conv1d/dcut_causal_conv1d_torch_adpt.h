#ifndef DCUT_CAUSAL_CONV1D_TORCH_ADPT_H
#define DCUT_CAUSAL_CONV1D_TORCH_ADPT_H

namespace vllm_ascend {

at::Tensor npu_dcut_causal_conv1d(const at::Tensor& output, const at::Tensor& x, const at::Tensor& weight,
                                  const at::Tensor& conv_state, const c10::optional<at::Tensor>& bias,
                                  const c10::optional<at::Tensor>& query_start_loc,
                                  const c10::optional<at::Tensor>& cache_indices,
                                  const c10::optional<at::Tensor>& state_offsets, int64_t activation_mode,
                                  int64_t pad_slot_id) {
  TORCH_CHECK(query_start_loc.has_value(), "query_start_loc cannot be empty.");
  TORCH_CHECK(cache_indices.has_value(), "cache_indices cannot be empty.");
  TORCH_CHECK(state_offsets.has_value(), "state_offsets cannot be empty.");

  EXEC_NPU_CMD(aclnnDcutCausalConv1d, x, weight, bias, conv_state, query_start_loc, cache_indices, c10::nullopt,
               state_offsets, activation_mode, pad_slot_id, 1, output);
  return output;
}

}  // namespace vllm_ascend
#endif
