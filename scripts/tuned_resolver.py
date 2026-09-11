# Copyright 2026 FlagOS Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Select and cache a launch config using LibTuner's key, caches and policy.

Cold selection benchmarks isolated tensor views; it does not perform the final
business launch. Unsupported cases return a reason; errors propagate.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import export_tuned_table as _export  # noqa: E402  (same directory)
from export_tuned_table import kernel_identity, unwrap_to_jit_function  # noqa: E402

_tuner_index: Dict[Tuple[str, str, Optional[str]], Any] = {}


def _same_file(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except OSError:
        return a == b


def _module_is_registered(tuner: Any, package_name: str) -> bool:
    """True when the tuner's kernel comes from a module imported as part of the
    package (``flag_gems.ops.mm``), as opposed to a copy of the same file that
    libtriton_jit loaded on its own for launching (module ``mm``, absent from
    ``sys.modules``). Both wrap the same kernel; the package's copy is the one
    the Python side tunes with."""
    identity = kernel_identity(tuner)
    jit_fn, _ = unwrap_to_jit_function(tuner.fn)
    py_fn = getattr(jit_fn, "fn", jit_fn)
    module_name = getattr(getattr(tuner, "base_fn", py_fn), "__module__", "") or ""
    module = sys.modules.get(module_name)
    if module is None:
        return False
    module_file = getattr(module, "__file__", None)
    return bool(module_file) and _same_file(module_file, identity["source_path"])


def find_tuner(
    package_name: str,
    kernel_id: str,
    source_path: Optional[str] = None,
    walk_ops: bool = True,
) -> Any:
    """The LibTuner whose kernel is the Triton function `kernel_id`.

    `source_path` (the .py file the C++ side launches from) disambiguates
    kernels that share a name across files (FlagGems' per-arch overrides);
    among copies of the same file the one imported under the package wins.
    """
    key = (
        package_name,
        kernel_id,
        os.path.realpath(source_path) if source_path else None,
    )
    if key in _tuner_index:
        return _tuner_index[key]
    _export.import_package_tree(package_name, walk_ops=walk_ops)
    libentry = _export.libentry_module_for(package_name)
    matches = [
        t
        for t in _export.find_tuners(libentry)
        if _export.kernel_identity(t)["kernel_id"] == kernel_id
    ]
    if source_path:
        matches = [
            t
            for t in matches
            if _same_file(_export.kernel_identity(t)["source_path"], source_path)
        ]
    if not matches:
        where = f" in {source_path}" if source_path else ""
        raise LookupError(
            f"no LibTuner in {package_name} wraps a kernel named '{kernel_id}'{where}"
        )
    if len(matches) > 1:
        preferred = [t for t in matches if _module_is_registered(t, package_name)]
        if preferred:
            matches = preferred
    if len(matches) > 1:
        files = {
            os.path.realpath(_export.kernel_identity(t)["source_path"]) for t in matches
        }
        if len(files) > 1:
            raise LookupError(
                f"kernel '{kernel_id}' is defined in several files of {package_name}: "
                + ", ".join(sorted(files))
                + "; pass source_path (ResolveArgs.source_path on the C++ side)"
            )
        # identical copies of one file (loaded more than once): any of them will do
    _tuner_index[key] = matches[0]
    return matches[0]


def build_grid(
    grid: Any, tuner: Any, args: Sequence[Any], kwargs: Dict[str, Any]
) -> Any:
    """Turn the caller's grid description into what the Triton autotuner needs.

    Accepts a callable (used as is), a tuple/list (constant grid), or a Python
    expression string evaluated with the kernel's argument names, `META`
    (the candidate config's kwargs), `triton` and `math` in scope, e.g.
    ``"(triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)"``.
    """
    if grid is None or callable(grid):
        return grid
    if isinstance(grid, (tuple, list)):
        return tuple(grid)
    if isinstance(grid, str):
        scope: Dict[str, Any] = {"math": math}
        try:
            import triton

            scope["triton"] = triton
        except ImportError:  # the expression may not need it
            pass
        scope.update(dict(zip(getattr(tuner, "arg_names", []), args)))
        scope.update({k: v for k, v in kwargs.items() if k not in ("grid", "warmup")})
        code = compile(grid, "<tuned_resolver grid>", "eval")

        def grid_fn(META):
            result = eval(
                code, {"__builtins__": {}}, {**scope, "META": META}
            )  # noqa: S307 - caller-supplied expression
            return (
                tuple(result) if isinstance(result, (tuple, list)) else (int(result),)
            )

        return grid_fn
    raise TypeError(
        f"grid must be a callable, a tuple, or an expression string, not {type(grid).__name__}"
    )


class _NeedsGrid(Exception):
    """Raised by _select_config when a benchmark is required but no grid was given."""


class _UnsupportedBenchmark(Exception):
    pass


def _local_tuner(tuner):
    """Own transient call state; share only the tuner's persistent caches/JIT.

    In particular, ordinary Python run() can continue using the original nargs.
    Triton's built-in reset/restore closures capture the original self, so they
    are rebuilt here against this invocation's state.
    """
    local = copy.copy(tuner)
    for name, value in vars(tuner).items():
        if isinstance(value, types.MethodType) and value.__self__ is tuner:
            setattr(local, name, types.MethodType(value.__func__, local))
    local.configs = list(tuner.configs)
    local.nargs = None
    if getattr(tuner, "user_defined_pre_hook", False) or getattr(
        tuner, "user_defined_post_hook", False
    ):
        raise _UnsupportedBenchmark(
            "custom tuner hooks require a consumer launch hook and are unsupported"
        )
    if getattr(tuner, "shared_config_pre_hook", None) is not None:
        raise _UnsupportedBenchmark(
            "shared_config_pre_hook requires a consumer launch hook"
        )
    reset = tuple(getattr(tuner, "reset_to_zero", ()) or ())
    restore = tuple(getattr(tuner, "restore_value", ()) or ())
    if reset or restore:
        copies = {}

        def pre_hook(named, reset_only=False):
            for name in reset:
                named[name].zero_()
            if not reset_only:
                copies.clear()
                copies.update((name, named[name].clone()) for name in restore)

        def post_hook(named, exception=None):
            for name in restore:
                named[name].copy_(copies[name])
            copies.clear()

        local.pre_hook, local.post_hook = pre_hook, post_hook
    return local


def _clone_arguments(args, kwargs):
    """Copy backing storage once, retaining strides, offsets and aliases.

    A byte view avoids interpreting uninitialised output values and permits
    differently typed views of the same storage. All view construction is
    detached from autograd. A bounded budget prevents tiny slices of huge
    allocations from unexpectedly exhausting device memory.
    """
    try:
        import torch
    except ImportError:
        return tuple(args), dict(kwargs)
    budget = int(os.environ.get("TRITON_JIT_BENCH_MAX_BYTES", str(1024**3)))
    storages = {}
    used = 0

    def clone(value):
        nonlocal used
        if not torch.is_tensor(value):
            return value
        if (
            value.layout != torch.strided
            or value.is_conj()
            or value.is_neg()
            or value.is_quantized
        ):
            raise _UnsupportedBenchmark(
                "benchmark requires ordinary strided tensor views"
            )
        storage = value.untyped_storage()
        key = (value.device, storage._cdata)
        if key not in storages:
            size = storage.nbytes()
            used += size
            if used > budget:
                raise _UnsupportedBenchmark(
                    "benchmark storage copies exceed TRITON_JIT_BENCH_MAX_BYTES"
                )
            with torch.no_grad():
                raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(
                    storage, 0, (size,), (1,)
                )
                storages[key] = raw.clone().untyped_storage()
        with torch.no_grad():
            return torch.empty(0, dtype=value.dtype, device=value.device).set_(
                storages[key],
                value.storage_offset(),
                tuple(value.shape),
                tuple(value.stride()),
            )

    return tuple(clone(value) for value in args), {
        name: clone(value) for name, value in kwargs.items()
    }


def _select_config(
    tuner: Any,
    libcache: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    launch_kwargs: Optional[Dict[str, Any]] = None,
    clone_tensors: bool = True,
) -> Tuple[Any, Tuple[Any, ...]]:
    """The resolve half of LibTuner.run(): returns (config, normalised key).

    `launch_kwargs` (grid / warmup) are only needed when a benchmark has to
    run; without them a cache miss raises _NeedsGrid. When benchmarking,
    tensor arguments are cloned first so the tuner's trial launches never
    touch the caller's live buffers.
    """
    tuner = _local_tuner(tuner)
    tuner.nargs = dict(zip(tuner.arg_names, args))
    try:
        all_args = {**tuner.nargs, **kwargs}
        _args = {k: v for k, v in all_args.items() if k in tuner.arg_names}
        config_key = tuner.get_key(_args)
        configs = list(getattr(tuner, "configs", []) or [])
        if len(configs) <= 1:
            return (configs[0] if configs else None), config_key
        if config_key in tuner.cache:
            config = tuner.cache[config_key]
        else:
            if not launch_kwargs:
                raise _NeedsGrid()
            bench_args, isolated_kwargs = (
                _clone_arguments(args, kwargs)
                if clone_tensors
                else (tuple(args), dict(kwargs))
            )
            bench_kwargs = {**isolated_kwargs, **launch_kwargs}
            tuner.nargs = dict(zip(tuner.arg_names, bench_args))
            if hasattr(tuner, "get_benchmark_key"):
                benchmark_key = tuner.get_benchmark_key(_args)
            else:
                benchmark_key = config_key
            bench_cache = libcache[tuner.benchmark_table_name, benchmark_key]
            pruned = tuner.prune_configs(bench_kwargs)

            def bench(config: Any) -> List[float]:
                ret = bench_cache.get(config)
                if ret is None:
                    ret = tuner._bench(*bench_args, config=config, **bench_kwargs)
                    if isinstance(ret, (int, float)):
                        ret = (ret, ret, ret)
                    if ret and all(math.isfinite(float(value)) for value in ret):
                        bench_cache[config] = tuple(ret)
                return list(ret)

            best_config, _timings = tuner.policy(
                bench, pruned, tuple(bench_args), bench_kwargs
            )
            tuner.cache[config_key] = best_config
            config = tuner.cache[config_key]
            full_nargs = {**tuner.nargs, **isolated_kwargs, **config.all_kwargs()}
            tuner.pre_hook(full_nargs, reset_only=True)
        # a config read back from the database has no pre_hook; recover the original
        if getattr(config, "pre_hook", None) is None:
            cached_kwargs = config.all_kwargs()
            for original in configs:
                if original.all_kwargs() == cached_kwargs:
                    config = original
                    break
        tuner.best_config = config
        return config, config_key
    finally:
        tuner.nargs = None


def resolve_with_tuner(
    tuner: Any,
    libentry_module: Any,
    args: Sequence[Any],
    kwargs: Dict[str, Any],
    grid: Any = None,
    clone_tensors: bool = True,
) -> Dict[str, Any]:
    """Resolve and describe the configuration for one launch.

    Returns a dict with kernel_id, key_columns ([{name, strategy}]), dtype_keys,
    key (normalised, as the tuner stores it), num_warps, num_stages, extra
    ({name: str}) and kwargs ([[name, value], ...] in kernel parameter order),
    or {"kernel_id": ..., "unsupported": reason}. `grid` is needed only when
    the key has not been tuned yet (see build_grid); without it such a key is
    reported as unsupported rather than benchmarked.
    """
    identity = _export.kernel_identity(tuner)
    kernel_id = identity["kernel_id"]
    reason = _export.refusal_reason(tuner, identity["wrappers"])
    if reason is not None:
        return {"kernel_id": kernel_id, "unsupported": reason}
    libcache = getattr(libentry_module, "libcache")
    blas_dialect = _export.is_flagblas(libentry_module)
    names = _export.strategy_names(tuner, libentry_module, blas_dialect)
    keys = list(getattr(tuner, "keys", []) or [])
    launch_kwargs = None
    grid_fn = build_grid(grid, tuner, args, kwargs)
    if grid_fn is not None:
        launch_kwargs = {"grid": grid_fn, "warmup": False}
    try:
        # Participating libraries share this lock with native _bench. It also
        # protects scratch allocation from another tuner's graph capture.
        with getattr(libentry_module, "benchmark_lock", nullcontext()):
            config, config_key = _select_config(
                tuner, libcache, args, kwargs, launch_kwargs, clone_tensors
            )
    except _UnsupportedBenchmark as error:
        return {"kernel_id": kernel_id, "unsupported": str(error)}
    except _NeedsGrid:
        return {
            "kernel_id": kernel_id,
            "unsupported": "this key has not been tuned yet and the caller supplied no launch grid; "
            "tune it from Python first or pass ResolveArgs.grid",
        }
    if config is None:
        return {
            "kernel_id": kernel_id,
            "unsupported": "the tuner has no candidate configs",
        }
    if getattr(config, "pre_hook", None) is not None:
        return {
            "kernel_id": kernel_id,
            "unsupported": "the selected config carries a pre_hook",
        }
    arg_order = {name: index for index, name in enumerate(identity["arg_names"])}
    types = _export.kwarg_types(tuner)
    kw_pairs: List[Tuple[str, Any]] = []
    for name, value in config.kwargs.items():
        if name not in arg_order:
            return {
                "kernel_id": kernel_id,
                "unsupported": f"constexpr '{name}' is not a kernel parameter",
            }
        if not isinstance(value, (bool, int, float, str)):
            return {
                "kernel_id": kernel_id,
                "unsupported": f"constexpr '{name}' has a non-serialisable value {value!r}",
            }
        kw_pairs.append(
            (name, _export.coerce_value(value, types.get(name, type(value))))
        )
    kw_pairs.sort(key=lambda item: arg_order[item[0]])
    extra: Dict[str, str] = {}
    for field in sorted(
        _export.triton_config_fields() - {"num_warps", "num_stages", "pre_hook"}
    ):
        value = getattr(config, field, None)
        if value is None or (field == "num_ctas" and value == 1):
            continue
        extra[field] = str(value)
    normalised = []
    for element in config_key:
        if isinstance(element, float) and element.is_integer():
            element = int(element)
        normalised.append(element)
    return {
        "kernel_id": kernel_id,
        "op_name": getattr(tuner, "__name__", kernel_id),
        "source_path": identity["source_path"],
        "key_columns": [
            {"name": key, "strategy": strategy} for key, strategy in zip(keys, names)
        ],
        "dtype_keys": max(0, len(config_key) - len(keys)),
        "key": normalised,
        "num_warps": int(getattr(config, "num_warps", 4)),
        "num_stages": int(getattr(config, "num_stages", 3)),
        "extra": extra,
        "kwargs": [[name, value] for name, value in kw_pairs],
    }


def resolve(
    package_name: str,
    kernel_id: str,
    args: Sequence[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    source_path: Optional[str] = None,
    grid: Any = None,
    clone_tensors: bool = True,
) -> Dict[str, Any]:
    """Entry point for the C++ bridge: find the tuner and resolve.

    `source_path` is the kernel file the caller launches from; `grid` is a
    Python expression, tuple or callable used only if the key must be tuned
    now (see build_grid).
    """
    tuner = find_tuner(package_name, kernel_id, source_path=source_path)
    libentry = _export.libentry_module_for(package_name)
    return resolve_with_tuner(
        tuner,
        libentry,
        list(args),
        dict(kwargs or {}),
        grid=grid,
        clone_tensors=clone_tensors,
    )


__all__ = ["build_grid", "find_tuner", "resolve", "resolve_with_tuner"]
