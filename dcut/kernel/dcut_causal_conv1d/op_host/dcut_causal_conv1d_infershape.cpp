// SPDX-License-Identifier: Apache-2.0

#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

using namespace ge;

namespace ops {
namespace {
constexpr int64_t TENSOR_INDEX = 0;

ge::graphStatus InferShapeDcutCausalConv1d(
    gert::InferShapeContext* context) {
  const gert::Shape* x_shape = context->GetInputShape(TENSOR_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);

  gert::Shape* y_shape = context->GetOutputShape(TENSOR_INDEX);
  OP_CHECK_NULL_WITH_CONTEXT(context, y_shape);
  *y_shape = *x_shape;
  return GRAPH_SUCCESS;
}
}  // namespace

IMPL_OP_INFERSHAPE(DcutCausalConv1d)
    .InferShape(InferShapeDcutCausalConv1d);
}  // namespace ops
