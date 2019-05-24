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
    static const PackedFunc* f_load = Registry::Get("module._LoadFromFile");
    Module m_so = (*f_load)(name + ".so", "");
    Module m_ptx = (*f_load)(name + ".ptx", "");
    m_so.Import(m_ptx);
    func = m_so.GetFunction("__tvm_main__", false);
  }

  TVMArgs Prepare(const std::vector<mxnet::NDArray> &args) {
    type_codes.resize(args.size());
    values.resize(args.size());
    for (int i = 0; i < static_cast<int>(args.size()); ++i) {
      type_codes[i] = kArrayHandle;
      values[i].v_handle = const_cast<DLTensor*>(&(args[i].data().dltensor()));
    }
    return TVMArgs(&values[0], &type_codes[0], args.size());
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
  *dispatch_mode = DispatchMode::kFComputeEx;
  in_attrs->at(0) = kDefaultStorage;
  in_attrs->at(1) = kDefaultStorage;
  out_attrs->at(0) = kDefaultStorage;
  return true;
}

static void TVMVectorAddStorageComputeExGPU(const nnvm::NodeAttrs& attrs,
                                            const mxnet::OpContext& ctx,
                                            const std::vector<NDArray>& inputs,
                                            const std::vector<OpReqType>& req,
                                            const std::vector<NDArray>& outputs) {
  using tvm::runtime::Registry;
  using tvm::runtime::PackedFunc;
  using tvm::runtime::TVMArgs;
  using tvm::runtime::TVMRetValue;
  using tvm::runtime::TVMCUDAFunctor;

  static thread_local TVMCUDAFunctor func("/home/ubuntu/Projects/tvm-compiler/build/myadd");
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  TVMArgs args = func.Prepare({inputs[0], inputs[1], outputs[0]});
  TVMRetValue rv;
  void* stream = static_cast<void*>(ctx.run_ctx.get_stream<gpu>()->stream_);
  int dev_type = kDLGPU;
  int dev_id = ctx.run_ctx.ctx.dev_id;
  TVMSetStream(dev_type, dev_id, stream);
  func.func.CallPacked(args, &rv);
  TVMSetStream(dev_type, dev_id, nullptr);
}
}  // namespace op
}  // namespace mxnet

NNVM_REGISTER_OP(tvm_vector_add)
.set_num_inputs(2)
.set_num_outputs(1)
.add_argument("a", "NDArray-or-Symbol", "first input")
.add_argument("b", "NDArray-or-Symbol", "second input")
.set_attr<mxnet::FInferShape>("FInferShape", mxnet::op::ElemwiseShape<2, 1>)
.set_attr<nnvm::FInferType>("FInferType", mxnet::op::ElemwiseType<2, 1>)
.set_attr<mxnet::FInferStorageType>("FInferStorageType", mxnet::op::TVMVectorAddStorageType)
.set_attr<mxnet::FComputeEx>("FComputeEx<gpu>", mxnet::op::TVMVectorAddStorageComputeExGPU);
