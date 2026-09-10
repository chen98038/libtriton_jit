# Autotune configurations for C++ launches

`TunedTable` supplies launch configurations. The caller still computes the grid
and launches its kernel. A configuration hit does not imply that the compiled
program has been loaded in this process.

## Offline tables

Export configurations from the intended LibTuner package and device:

```sh
python scripts/export_tuned_table.py --package flag_blas --kernel 'sgemv_*' \
  --cache-namespace flag_blas/sgemv@revision --output tuned.json
```

`--list` lists discovered tuners. If a kernel name exists in multiple files,
use `--source /path/to/kernel.py`. An explicit namespace must identify the same
immutable source version at export and lookup; otherwise the exporter uses the
source path. The runtime does not hash the full source dependency graph.

For a kernel keyed by `(m, n)` and three tensor dtypes:

```cpp
#include "triton_jit/tuned_config.h"

auto& table = triton_jit::TunedTable::instance();
table.load("tuned.json", device_index);
const auto id = triton_jit::scoped_kernel_id(
    "sgemv_n_kernel", "flag_blas/sgemv@revision");

const int64_t dims[] = {m, n};
const char* dtypes[] = {"torch.float32", "torch.float32", "torch.float32"};
const auto* config = table.find(id, device_index, {dims, 2, dtypes, 3});
if (config) {
  // Use config->kwargs and config->to_compile_options<CompileOptions>() at launch.
} else {
  // Use the caller's existing fallback.
}
```

The key columns and dtypes must match that kernel's exported schema. Reuse the
bound `id` on hot calls. `find()` allocates nothing, is `noexcept`, and never
enters Python. `load_from_env(device)` reads colon-separated paths from
`TRITON_JIT_TUNED_TABLE` and reports invalid files without throwing.

Tables validate backend and device identity. `TRITON_JIT_TUNED_IGNORE_DEVICE=1`
explicitly bypasses the device-name check; `TRITON_JIT_LOG_TUNED=1` enables load
logging. Key strategies (`default`, `log`, `align32`) match the Python tuner.
Unsupported hooks, heuristics and non-serializable configs are recorded as
unsupported and miss at lookup.

New exports use format v2 with a source namespace. This reader also accepts v1
for legacy unscoped lookups; source-scoped lookups do not silently trust v1
records. Old readers reject v2. A namespace is a consumer-managed version
contract, not automatic proof that source contents are unchanged.

Table/config pointers remain valid until `clear()`. Loading creates a new table
generation; existing handles retain their previous generation. `clear()` is only
for quiescent teardown or tests, never concurrent with lookup or resolution.

## Preparing programs and freezing cold work

Prepare with the same signature, compile options and device as the normal launch:

```cpp
function.prepare(signature, options, device_index, stream);
if (!function.is_prepared(signature, options, device_index)) {
  throw std::runtime_error("program is not prepared");
}
```

Generate `signature` with the normal argument-handling path; a tuned key can
cover multiple launch signatures. `prepare()` compiles and loads the GPU module
without launching the business kernel or modifying its tensors.

`ScopedFreeze` is a process-wide, nestable policy: ready calls can execute, while
cold work throws `FrozenMissError`. Without an explicit guard, supported backends
also check whether the supplied launch stream is being captured. The error's
`work()` distinguishes missing config, program or function; `restriction()`
distinguishes explicit freeze from capture.

Prepare before capture. Compilation or tuning may synchronize or start nested
capture and is unsuitable inside capture. CPU/Python code is not recorded as
ordinary GPU graph nodes. After capture, graph replay does not re-run the
original per-operator configuration lookup. Eager work can run before or after
replay; keeping a global freeze active also forbids cold preparation there.

Program and function caches use short locks; compilation does not hold those
locks. Concurrent misses may compile redundantly, but publication retains a
completed program and backend loading is synchronized.

## Optional online resolver

Link `TritonJIT::triton_jit_torch_resolver`, enabled by
`TRITON_JIT_BUILD_TORCH_RESOLVER` (default ON). Only that target adds a dependency
on `libtorch_python`; the core runtime already uses Torch and an embedded Python
compiler. Its hot configuration lookup does not enter Python.

Install the resolver once, then describe the actual kernel arguments:

```cpp
#include "triton_jit/torch/tuned_resolver.h"

auto& table = triton_jit::TunedTable::instance();
table.set_resolver(triton_jit::torch_resolver::make_resolver({"flag_gems"}));

triton_jit::torch_resolver::ResolveArgs ctx;
ctx.source_path = kernel_source;
ctx.cache_namespace = "flag_gems/mm@revision";
ctx.args = kernel_arguments;  // Match the Python kernel's positional parameters.
ctx.grid = "(triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)";
const auto id = triton_jit::scoped_kernel_id(kernel_name, ctx.cache_namespace);
const auto* config = table.find(id, device_index, key);
if (!config) config = table.resolve(kernel_name, device_index, key, &ctx, stream);
```

The namespace must match offline exports. Without an explicit namespace, the
adapter uses `source_path`. `find_for_context()` performs the same binding but
constructs the identity string on each call; callers may pre-bind `id` above.

`resolve()` returns a configuration, not a calculation result. A C++ hit returns
directly. A cold miss calls the resolver outside runtime locks; Python may reuse
its config cache or benchmark candidates using LibTuner's own pruning and policy.
The result is validated and stored under the normalized key. Program preparation
and the final business launch are separate steps.

No resolver or a declared unsupported case returns no configuration. Without
`grid`, an untuned key is unsupported; cached selections remain usable. Grid
expressions have kernel arguments, `META`, `triton` and `math` in scope, without
Python builtins. Other errors propagate to callers. Frozen/captured cold misses
are rejected before the callback runs.

### Concurrency and benchmark isolation

- Pending requests share a result per identity/device/key. Once the schema is
  bound, keys are normalized before sharing; the first unknown-schema requests
  can still tune separately. Misses and exceptions are not retained as configs.
- Online entries are append-only with stable addresses. Publication locks cover
  reading the latest state, validation and insertion; Python runs outside them.
  Existing handles see later online entries in their generation, while `info()`
  is metadata from when that handle was published.
- The Torch wait hook releases a held GIL. Custom Python-facing resolvers must
  provide an equivalent wait hook; identity hooks must not enter Python.
- Cold benchmarking copies each backing storage once, preserving dtype, shape,
  stride, offset and aliases across args/kwargs. Local tuner state and built-in
  reset/restore hooks use those copies. Persistent caches and JIT objects remain
  shared. The final business launch uses the caller's original arguments.
- `TRITON_JIT_BENCH_MAX_BYTES` bounds copied storage (default 1 GiB). Unsupported
  layouts, quantized/special views and custom launch hooks are rejected. The
  implementation does not promise arbitrary third-party hook/policy safety.
- The CUDA adapter checks tensor devices and uses the supplied launch stream,
  restoring the previous stream afterwards. Cross-stream input dependencies
  remain the caller's responsibility.

Mixed native Python/C++ cold tuning needs consumer-side coordination. Historical
FlagGems experiments used a shared benchmark RLock and
`FLAGGEMS_BENCHMARK_MODE=event` set **before import**. Those changes belong in
FlagGems and are not applied by this library. A benchmark lock alone does not
make default global graph benchmarking safe alongside arbitrary CUDA work.

## Validation

Run the normal CTest suite for CPU-capable regressions. See
[CUDA test commands](../tests/cuda/README.md) to enable the real GIL and
capture/replay checks, or run tensor layouts under compute-sanitizer.
