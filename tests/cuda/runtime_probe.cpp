#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "triton_jit/triton_jit_function.h"
#include "triton_jit/torch/tuned_resolver.h"
#include "triton_jit/freeze.h"
#include <memory>
namespace py = pybind11;
using namespace triton_jit;

PYBIND11_MODULE(tuned_runtime_probe, m) {
  m.def("install", [](const std::string& path) {
    auto& table=TunedTable::instance(); table.clear();
    table.set_resolver(torch_resolver::make_resolver({"fake_package", "fake_tuned_resolver", path}));
  });
  m.def("resolve", [](const std::string& name,const std::string& source,bool frozen,uintptr_t stream) {
    torch_resolver::ResolveArgs ctx; ctx.source_path=source;
    const int64_t dims[]={1000,4000}; const char* dtypes[]={"torch.float16","torch.bfloat16"};
    std::unique_ptr<ScopedFreeze> guard;
    if (frozen) guard=std::make_unique<ScopedFreeze>();
    const auto* cfg=TunedTable::instance().resolve(name,0,{dims,2,dtypes,2},&ctx,reinterpret_cast<void*>(stream));
    py::dict result;
    if (cfg) {
      result["source"]=*cfg->get_str("SRC");
      result["calls"]=cfg->get_i64("CALLS",-1);
    }
    return result;
  });
  m.def("launch", [](const std::string& path,uintptr_t x,uintptr_t y,int n,int block,uintptr_t stream,bool frozen) {
    std::unique_ptr<ScopedFreeze> guard;
    if (frozen) guard=std::make_unique<ScopedFreeze>();
    auto& f=TritonJITFunction::get_instance(path,"vector_kernel");
    f(reinterpret_cast<CUstream>(stream),(n+block-1)/block,1,1,4,1,
      device_ptr<float>(reinterpret_cast<float*>(x)),device_ptr<float>(reinterpret_cast<float*>(y)),n,block);
  });
  m.def("prepare", [](const std::string& path,uintptr_t x,uintptr_t y,int n,int block) {
    auto& f=TritonJITFunction::get_instance(path,"vector_kernel");
    ParameterBuffer buffer; c10::SmallVector<std::string> signature;
    ArgHandle handler={f.get_static_sig(),buffer,signature,0};
    handler.handle_args(device_ptr<float>(reinterpret_cast<float*>(x)),device_ptr<float>(reinterpret_cast<float*>(y)),n,block);
    CompileOptions opts;opts.num_warps=4;opts.num_stages=1;
    f.prepare(join_sig(signature),opts,0);
    return f.is_prepared(join_sig(signature),opts,0);
  });
}
