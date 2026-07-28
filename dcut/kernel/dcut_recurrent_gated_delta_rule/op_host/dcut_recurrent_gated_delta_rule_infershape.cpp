/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace gert;
namespace ops {
namespace {
constexpr size_t VALUE_INDEX = 2;
constexpr size_t STATE_INDEX = 4;
constexpr size_t VALUE_DIM = 3;
constexpr size_t STATE_DIM = 4;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;

ge::graphStatus InferShapeDcutRecurrentGatedDeltaRule(
    InferShapeContext* context) {
  if (context == nullptr) {
    OP_LOGE("DcutRecurrentGatedDeltaRule", "inference context is null");
    return ge::GRAPH_FAILED;
  }

  auto op_name = context->GetNodeName();
  auto shape_value = context->GetInputShape(VALUE_INDEX);
  auto shape_initial_state = context->GetInputShape(STATE_INDEX);
  auto shape_out = context->GetOutputShape(DIM_0);
  auto shape_final_state = context->GetOutputShape(DIM_1);
  if (shape_value == nullptr || shape_initial_state == nullptr ||
      shape_out == nullptr || shape_final_state == nullptr) {
    OP_LOGE(op_name, "[InferShape] shape is null");
    return ge::GRAPH_FAILED;
  }

  shape_out->SetDimNum(VALUE_DIM);
  shape_out->SetDim(DIM_0, shape_value->GetDim(DIM_0));
  shape_out->SetDim(DIM_1, shape_value->GetDim(DIM_1));
  shape_out->SetDim(DIM_2, shape_value->GetDim(DIM_2));

  shape_final_state->SetDimNum(STATE_DIM);
  shape_final_state->SetDim(DIM_0, shape_initial_state->GetDim(DIM_0));
  shape_final_state->SetDim(DIM_1, shape_initial_state->GetDim(DIM_1));
  shape_final_state->SetDim(DIM_2, shape_initial_state->GetDim(DIM_2));
  shape_final_state->SetDim(DIM_3, shape_initial_state->GetDim(DIM_3));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeDcutRecurrentGatedDeltaRule(
    InferDataTypeContext* context) {
  context->SetOutputDataType(DIM_0, ge::DT_BF16);
  context->SetOutputDataType(DIM_1, ge::DT_BF16);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(DcutRecurrentGatedDeltaRule)
    .InferShape(InferShapeDcutRecurrentGatedDeltaRule)
    .InferDataType(InferDataTypeDcutRecurrentGatedDeltaRule);
}  // namespace ops
