// SPDX-License-Identifier: Apache-2.0

#ifndef DCUT_ACLNN_RECURRENT_GATED_DELTA_RULE_H
#define DCUT_ACLNN_RECURRENT_GATED_DELTA_RULE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus
aclnnDcutRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value,
    const aclTensor* beta, aclTensor* state_ref,
    const aclTensor* actual_seq_lengths, const aclTensor* ssm_state_indices,
    const aclTensor* g, const aclTensor* gk,
    const aclTensor* num_accepted_tokens, float scale_value, aclTensor* out,
    uint64_t* workspace_size, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus
aclnnDcutRecurrentGatedDeltaRule(void* workspace, uint64_t workspace_size,
                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // DCUT_ACLNN_RECURRENT_GATED_DELTA_RULE_H
