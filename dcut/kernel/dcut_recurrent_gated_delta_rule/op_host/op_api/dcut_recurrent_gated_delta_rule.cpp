// SPDX-License-Identifier: Apache-2.0

#include "../dcut_recurrent_gated_delta_rule.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(DcutRecurrentGatedDeltaRule);

const aclTensor* DcutRecurrentGatedDeltaRule(
    const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* beta, aclTensor* state_ref,
    const aclTensor* actual_seq_lengths, const aclTensor* ssm_state_indices,
    const aclTensor* g, const aclTensor* gk,
    const aclTensor* num_accepted_tokens, float scale_value,
    aclOpExecutor* executor) {
  L0_DFX(DcutRecurrentGatedDeltaRule, query, key, value, beta, state_ref,
         actual_seq_lengths, ssm_state_indices, g, gk, num_accepted_tokens,
         scale_value);

  DataType out_type = DataType::DT_BF16;
  Format format = Format::FORMAT_ND;
  auto out = executor->AllocTensor(out_type, format, format);
  OP_CHECK(out != nullptr,
           OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "out AllocTensor failed."),
           return nullptr);

  auto ret = INFER_SHAPE(
      DcutRecurrentGatedDeltaRule,
      OP_INPUT(query, key, value, beta, state_ref, actual_seq_lengths,
               ssm_state_indices, g, gk, num_accepted_tokens),
      OP_OUTPUT(out, state_ref), OP_ATTR(scale_value));
  OP_CHECK_INFERSHAPE(ret != ACLNN_SUCCESS, return nullptr,
                      "DcutRecurrentGatedDeltaRule InferShape failed.");

  ret = ADD_TO_LAUNCHER_LIST_AICORE(
      DcutRecurrentGatedDeltaRule,
      OP_INPUT(query, key, value, beta, state_ref, actual_seq_lengths,
               ssm_state_indices, g, gk, num_accepted_tokens),
      OP_OUTPUT(out, state_ref), OP_ATTR(scale_value));
  OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
      ret != ACLNN_SUCCESS, return nullptr,
      "DcutRecurrentGatedDeltaRule ADD_TO_LAUNCHER_LIST_AICORE failed.");
  return out;
}

}  // namespace l0op
