/*!
 * Copyright (c) 2020 by Contributors
 * \file src/op/declare/annotation.cc
 * \brief Declaration of annotation operators
 */
#include "mnm/op.h"
#include "mnm/tensor.h"
#include "../schema/annotation.h"

namespace mnm {
namespace op {
namespace declare {

using namespace mnm::op::schema;
using namespace mnm::value;

MNM_OP_DECLARE("mnm.op.compiler_begin", [](const CallValues& call) {
  const auto* args = call->args.as<CompilerArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::string compiler = args->compiler;

  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>(x->shape, x->shape + x->ndim));
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

MNM_OP_DECLARE("mnm.op.compiler_end", [](const CallValues& call) {
  const auto* args = call->args.as<CompilerArgs>();
  CHECK(args != nullptr);
  DLTensor* x = args->x;
  std::string compiler = args->compiler;

  call->out = TensorValue::Assemble(/*ctx=*/x->ctx,
                                    /*dtype=*/x->dtype,
                                    /*shape=*/std::vector<int64_t>(x->shape, x->shape + x->ndim));
  call->ctx = x->ctx;
}).set_attr<TOpPattern>("TOpPattern", kOpaque);

}  // namespace declare
}  // namespace op
}  // namespace mnm