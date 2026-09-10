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
