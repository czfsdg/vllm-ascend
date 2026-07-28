// SPDX-License-Identifier: Apache-2.0

#include "aclnn_dcut_recurrent_gated_delta_rule.h"
#include "../dcut_recurrent_gated_delta_rule.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

using namespace op;

namespace {

struct DcutRecurrentGatedDeltaRuleParams {
  const aclTensor* query = nullptr;
  const aclTensor* key = nullptr;
  const aclTensor* value = nullptr;
  const aclTensor* beta = nullptr;
  const aclTensor* state = nullptr;
  const aclTensor* actual_seq_lengths = nullptr;
  const aclTensor* ssm_state_indices = nullptr;
  const aclTensor* g = nullptr;
  const aclTensor* gk = nullptr;
  const aclTensor* num_accepted_tokens = nullptr;
  float scale = 1.0F;
  const aclTensor* out = nullptr;
};

const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16};
const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT};
const std::initializer_list<op::DataType> BETA_TYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16};
const std::initializer_list<op::DataType> SEQ_LENS_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32};
const std::initializer_list<op::DataType> SSM_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32};
const std::initializer_list<op::DataType> G_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT};
const std::initializer_list<op::DataType> ACC_TO_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32};
const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16};

bool CheckNotNull(const DcutRecurrentGatedDeltaRuleParams& params) {
  OP_CHECK_NULL(params.query, return false);
  OP_CHECK_NULL(params.key, return false);
  OP_CHECK_NULL(params.value, return false);
  OP_CHECK_NULL(params.state, return false);
  OP_CHECK_NULL(params.beta, return false);
  OP_CHECK_NULL(params.actual_seq_lengths, return false);
  OP_CHECK_NULL(params.ssm_state_indices, return false);
  OP_CHECK_NULL(params.out, return false);
  return true;
}

bool CheckDtypeValid(const DcutRecurrentGatedDeltaRuleParams& params) {
  OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST,
                             return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST,
                             return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.state, STATE_TYPE_SUPPORT_LIST,
                             return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, BETA_TYPE_SUPPORT_LIST,
                             return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.actual_seq_lengths,
                             SEQ_LENS_TYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(params.ssm_state_indices, SSM_TYPE_SUPPORT_LIST,
                             return false);
  if (params.g != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
  }
  if (params.gk != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(params.gk, G_TYPE_SUPPORT_LIST, return false);
  }
  if (params.num_accepted_tokens != nullptr) {
    OP_CHECK_DTYPE_NOT_SUPPORT(params.num_accepted_tokens,
                               ACC_TO_TYPE_SUPPORT_LIST, return false);
  }
  OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
  return true;
}

aclnnStatus CheckParams(DcutRecurrentGatedDeltaRuleParams& params) {
  CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
  OP_LOGD("DcutRecurrentGatedDeltaRule check params success.");
  return ACLNN_SUCCESS;
}

aclnnStatus PreProcess(DcutRecurrentGatedDeltaRuleParams& params) {
  params.query->SetOriginalShape(params.query->GetViewShape());
  params.key->SetOriginalShape(params.key->GetViewShape());
  params.value->SetOriginalShape(params.value->GetViewShape());
  params.beta->SetOriginalShape(params.beta->GetViewShape());
  params.state->SetOriginalShape(params.state->GetViewShape());
  params.actual_seq_lengths->SetOriginalShape(
      params.actual_seq_lengths->GetViewShape());
  params.ssm_state_indices->SetOriginalShape(
      params.ssm_state_indices->GetViewShape());
  return ACLNN_SUCCESS;
}

}  // namespace

extern "C" {

aclnnStatus aclnnDcutRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* beta, aclTensor* state_ref,
    const aclTensor* actual_seq_lengths, const aclTensor* ssm_state_indices,
    const aclTensor* g, const aclTensor* gk,
    const aclTensor* num_accepted_tokens, float scale_value, aclTensor* out,
    uint64_t* workspace_size, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(
      aclnnDcutRecurrentGatedDeltaRule,
      DFX_IN(query, key, value, beta, state_ref, actual_seq_lengths,
             ssm_state_indices, g, gk, num_accepted_tokens, scale_value),
      DFX_OUT(out, state_ref));

  auto unique_executor = CREATE_EXECUTOR();
  CHECK_RET(unique_executor.get() != nullptr,
            ACLNN_ERR_INNER_CREATE_EXECUTOR);

  DcutRecurrentGatedDeltaRuleParams params{
      query, key, value, beta, state_ref, actual_seq_lengths,
      ssm_state_indices, g, gk, num_accepted_tokens, scale_value, out};
  CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
  auto ret = PreProcess(params);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  auto query_contiguous = l0op::Contiguous(query, unique_executor.get());
  auto key_contiguous = l0op::Contiguous(key, unique_executor.get());
  auto value_contiguous = l0op::Contiguous(value, unique_executor.get());
  auto beta_contiguous = l0op::Contiguous(beta, unique_executor.get());
  auto actual_seq_lengths_contiguous =
      l0op::Contiguous(actual_seq_lengths, unique_executor.get());
  auto ssm_state_indices_contiguous =
      l0op::Contiguous(ssm_state_indices, unique_executor.get());
  if (g != nullptr) {
    g = l0op::Contiguous(g, unique_executor.get());
  }
  if (gk != nullptr) {
    gk = l0op::Contiguous(gk, unique_executor.get());
  }
  if (num_accepted_tokens != nullptr) {
    num_accepted_tokens =
        l0op::Contiguous(num_accepted_tokens, unique_executor.get());
  }
  auto out_contiguous = l0op::Contiguous(out, unique_executor.get());

  auto out_ret = l0op::DcutRecurrentGatedDeltaRule(
      query_contiguous, key_contiguous, value_contiguous, beta_contiguous,
      state_ref, actual_seq_lengths_contiguous, ssm_state_indices_contiguous,
      g, gk, num_accepted_tokens, scale_value, unique_executor.get());
  if (out_ret == nullptr) {
    return ACLNN_ERR_INNER_NULLPTR;
  }
  auto view_copy_result =
      l0op::ViewCopy(out_ret, out_contiguous, unique_executor.get());
  if (view_copy_result == nullptr) {
    return ACLNN_ERR_INNER_NULLPTR;
  }

  *workspace_size = unique_executor->GetWorkspaceSize();
  unique_executor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnDcutRecurrentGatedDeltaRule(
    void* workspace, uint64_t workspace_size, aclOpExecutor* executor,
    aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnDcutRecurrentGatedDeltaRule);
  return CommonOpExecutorRun(workspace, workspace_size, executor, stream);
}

}  // extern "C"
