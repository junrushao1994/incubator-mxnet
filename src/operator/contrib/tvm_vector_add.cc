// #define TVM_RUNTIME_HEADER_ONLY 1
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <mxnet/base.h>
#include "../tensor/elemwise_binary_op.h"

namespace tvm {
namespace runtime {

class TVMCUDAFunctor {
 public:
  // constructor
  explicit TVMCUDAFunctor(const std::string& name) {
    static const PackedFunc* f_so = Registry::Get("module.loadfile_so");
    static const PackedFunc* f_ptx = Registry::Get("module.loadfile_ptx");
    Module m_so = (*f_so)(name + ".so");
    Module m_ptx = (*f_ptx)(name + ".ptx");
    m_so.Import(m_ptx);
    func = m_so.GetFunction("__tvm_main__", false);
  }

  void Prepare(const std::vector<mxnet::NDArray> &args) {
    type_codes.resize(args.size());
    values.resize(args.size());
    for (int i = 0; i < static_cast<int>(args.size()); ++i) {
      type_codes[i] = mxnet::kTVMNDArrayTypeCode;
      values[i].v_handle = const_cast<DLTensor*>(&(args[i].data().dltensor()));
    }
  }
 public:

  std::vector<TVMValue> values;
  std::vector<int> type_codes;
  PackedFunc func;
};

}  // namespace runtime
}  // namespace tvm

namespace mxnet {
namespace op {

static inline bool TVMVectorAddStorageType(const nnvm::NodeAttrs& attrs,
                               const int dev_mask,
                               DispatchMode* dispatch_mode,
                               std::vector<int> *in_attrs,
                               std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2);
  CHECK_EQ(out_attrs->size(), 1);
  return ElemwiseBinaryOp::PreferDenseStorageType<true, true, true>(
               attrs, dev_mask, dispatch_mode, in_attrs, out_attrs);
}

static void TVMVectorAddStorageComputeExGPU(const nnvm::NodeAttrs& attrs,
                                            const mxnet::OpContext& ctx,
                                            const std::vector<NDArray>& inputs,
                                            const std::vector<OpReqType>& req,
                                            const std::vector<NDArray>& outputs) {
  using namespace tvm::runtime;
  static const PackedFunc* f_set_stream = Registry::Get("_TVMSetStream");
  thread_local TVMCUDAFunctor func("myadd");
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  func.Prepare({inputs[0], inputs[1], outputs[0]});

  TVMRetValue rv;
  TVMArgs args(&func.values[0], &func.type_codes[0], 3);
  void* stream = ctx.run_ctx.stream;
  int dev_type = kDLGPU;
  int dev_id = ctx.run_ctx.ctx.dev_id;
  (*f_set_stream)(dev_type, dev_id, stream);
  func.func.CallPacked(args, &rv);
  (*f_set_stream)(dev_type, dev_id, nullptr);
}
}  // namespace op
}  // namespace mxnet

NNVM_REGISTER_OP(tvm_vector_add)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const mxnet::NodeAttrs& attrs) {
    return std::vector<std::string>{"a", "b"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", mxnet::op::ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
.set_attr<mxnet::FInferStorageType>("FInferStorageType", mxnet::op::TVMVectorAddStorageType)
.set_attr<mxnet::FComputeEx>("FComputeEx<cpu>", mxnet::op::TVMVectorAddStorageComputeExGPU)
.add_argument("a", "NDArray-or-Symbol", "first input")
.add_argument("b", "NDArray-or-Symbol", "second input");
