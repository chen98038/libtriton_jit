"""Bridge between the C++ autotuned_call path and FlagGems' Python LibTuner.

Called from C++ on cache miss. Responsibilities:
  1. Import the kernel's .py file.
  2. Walk the decorator chain LibEntry -> LibTuner -> Heuristics -> JITFunction
     to find the Autotuner.
  3. Call `LibTuner.resolve_config(*args, **kwargs)` to pick the autotuned
     `triton.Config` (this may trigger benchmarking on a fresh key).
  4. Walk further to find any `triton.runtime.Heuristics` layer; evaluate
     each heuristic with (current args + autotune constexprs) and merge
     into the kwargs set.
  5. Pull defaults from the JITFunction for any constexpr param not covered
     by autotune or heuristics.
  6. Reorder the merged dict to match the kernel's positional parameter
     order (the C++ side forwards `cfg.kwargs` to the kernel positionally).
  7. Normalize values to (int | bool); raise on float/other types (V0 scope).

Return shape (consumed by the C++ helper in Phase 4):

    {
        "num_warps": int,
        "num_stages": int,
        "kwargs": [(name, value), ...],   # value is int or bool
    }

V0 limitations (documented in `.claude/autotune_config_interface_design.md` §9
and `.claude/progress.md`):
  - Only int and bool constexpr values supported (Q11: float reserved).
  - `num_ctas` / `maxnreg` / warp-spec params not propagated (FlagGems
    bmm/baddbmm/max do not tune these in V0).
  - `pre_hook` not invoked from the C++ path (Q9: only group_gemm uses it
    and it has no C++ wrapper).
"""

from __future__ import annotations

import importlib.util
import inspect
import threading
from typing import Any, Dict, List, Tuple, Union

import triton

ConfigValue = Union[int, bool]


# (kernel_path, function_name) -> decorated kernel (typically a LibEntry instance)
_kernel_cache: Dict[Tuple[str, str], Any] = {}
_kernel_cache_lock = threading.Lock()


def _load_kernel(kernel_path: str, function_name: str) -> Any:
    """Import the module at `kernel_path` and return its `function_name` attr.

    Cached after first call to avoid repeated `importlib` work.
    """
    cache_key = (kernel_path, function_name)
    cached = _kernel_cache.get(cache_key)
    if cached is not None:
        return cached

    with _kernel_cache_lock:
        cached = _kernel_cache.get(cache_key)
        if cached is not None:
            return cached

        module_name = f"_autotune_bridge_kernel_{abs(hash(cache_key))}"
        spec = importlib.util.spec_from_file_location(module_name, kernel_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"autotune_bridge: cannot load kernel module from {kernel_path!r}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kernel = getattr(module, function_name, None)
        if kernel is None:
            raise RuntimeError(
                f"autotune_bridge: function {function_name!r} not found in "
                f"{kernel_path!r}"
            )

        _kernel_cache[cache_key] = kernel
        return kernel


def _find_autotuner(root: Any) -> triton.runtime.Autotuner:
    """Walk `.fn` chain from `root` to find the Autotuner layer."""
    fn = root
    while not isinstance(fn, triton.runtime.JITFunction):
        if isinstance(fn, triton.runtime.Autotuner):
            return fn
        fn = fn.fn
    raise RuntimeError(
        "autotune_bridge: no Autotuner found in the decorator chain — "
        "the kernel is not @libtuner-decorated, so there is no tuned Config "
        "to look up. Use the plain `operator()` path instead of autotuned_call."
    )


def _find_heuristics(root: Any) -> Union[triton.runtime.Heuristics, None]:
    """Walk `.fn` chain to find a Heuristics layer; None if absent."""
    fn = root
    while not isinstance(fn, triton.runtime.JITFunction):
        if isinstance(fn, triton.runtime.Heuristics):
            return fn
        fn = fn.fn
    return None


def _find_jit_function(root: Any) -> triton.runtime.JITFunction:
    """Walk `.fn` chain to the innermost JITFunction."""
    fn = root
    while not isinstance(fn, triton.runtime.JITFunction):
        fn = fn.fn
    return fn


def _coerce_for_cpp(name: str, value: Any) -> ConfigValue:
    """Validate and narrow a constexpr value to what triton_jit::Config carries.

    The C++ side uses `std::variant<int64_t, bool>`. Check bool before int,
    because in Python `bool` is a subclass of `int`.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    raise TypeError(
        f"autotune_bridge: constexpr {name!r} has unsupported type "
        f"{type(value).__name__} (value={value!r}); V0 supports int and bool "
        f"only (see Q11 in progress.md)"
    )


def get_tune_key_names(kernel_path: str, function_name: str) -> List[str]:
    """Return the LibTuner's `keys` list — the names of the args that
    participate in the autotune cache key (the `key=[...]` arg of `@libtuner`).

    The C++ AutotunedCall calls this once at construction time to assert
    that the positional `TuneKey` the caller assembles aligns with what
    Python uses — see Q4 in progress.md.

    Raises RuntimeError if no LibTuner in the chain.
    """
    root = _load_kernel(kernel_path, function_name)
    autotuner = _find_autotuner(root)
    # `keys` (plural) is set by LibTuner.__init__ at libentry.py:285.
    return list(autotuner.keys)


def _generic_resolve_config(at, args, kwargs):
    """Resolve a Config for a bare triton.runtime.Autotuner (no LibTuner /
    no resolve_config). Mirrors the resolve-half of Autotuner.run() -- every
    step EXCEPT the final self.fn.run() launch. Benchmark-time hooks
    (reset_to_zero / restore_value / config.pre_hook) run inside self._bench,
    exactly as in run(). dtype is appended to the (Python-side) key just like
    run() does, so bare-autotune caching is dtype-correct too.
    """
    at.nargs = dict(zip(at.arg_names, args))
    if len(at.configs) > 1:
        all_args = {**at.nargs, **kwargs}
        _args = {k: v for k, v in all_args.items() if k in at.arg_names}
        key = [_args[k] for k in at.keys if k in _args]
        for _, arg in _args.items():
            if hasattr(arg, "dtype"):
                key.append(str(arg.dtype))
        key = tuple(key)
        if key not in at.cache:
            pruned = at.prune_configs(kwargs)
            timings = {c: at._bench(*args, config=c, **kwargs) for c in pruned}
            at.cache[key] = min(timings, key=timings.get)
            full_nargs = {**at.nargs, **kwargs, **at.cache[key].all_kwargs()}
            at.pre_hook(full_nargs, reset_only=True)  # benchmark cleanup, Python-side
        config = at.cache[key]
    else:
        config = at.configs[0]
    at.best_config = config
    at.nargs = None
    return config


def _resolve_any(autotuner, args, kwargs):
    """Dispatch: LibTuner has its own resolve_config; a bare Autotuner uses
    the generic one above."""
    if hasattr(autotuner, "resolve_config"):
        return autotuner.resolve_config(*args, **kwargs)
    return _generic_resolve_config(autotuner, args, kwargs)


def lookup_config(
    kernel_path: str,
    function_name: str,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the autotuned Config for one kernel call.

    Args:
        kernel_path: absolute path to the kernel's .py source file.
        function_name: name of the LibEntry-wrapped kernel in that module.
        args: positional args of the kernel call, in the same prefix order
            that the C++ `autotuned_call` was invoked with (i.e., NOT
            including the trailing constexpr block — those are the unknowns
            we're resolving here).
        kwargs: keyword args of the call (FlagGems ops rarely use these).

    Returns:
        A dict with keys "num_warps", "num_stages", "kwargs" where "kwargs"
        is a list of (name, value) tuples in kernel positional order,
        suitable for direct construction of `triton_jit::Config` on the C++
        side.
    """
    root = _load_kernel(kernel_path, function_name)
    autotuner = _find_autotuner(root)
    jit_function = _find_jit_function(autotuner.fn)

    config = _resolve_any(autotuner, args, kwargs)
    merged: Dict[str, Any] = dict(config.kwargs)

    heuristics = _find_heuristics(autotuner.fn)
    if heuristics is not None:
        heur_input = {
            **dict(zip(heuristics.arg_names, args)),
            **kwargs,
            **merged,
        }
        for name, heur_fn in heuristics.values.items():
            merged[name] = heur_fn(heur_input)

    # Rebuild in kernel positional order, filling defaults for any constexpr
    # param the autotune + heuristics chain didn't cover. C++ forwards
    # cfg.kwargs positionally; the order MUST match the kernel signature.
    ordered: Dict[str, Any] = {}
    for p in jit_function.params:
        if not p.is_constexpr:
            continue
        if p.name in merged:
            ordered[p.name] = merged[p.name]
        elif p.default is not inspect.Parameter.empty:
            ordered[p.name] = p.default
        else:
            raise RuntimeError(
                f"autotune_bridge: constexpr param {p.name!r} of "
                f"{function_name!r} has no value from autotune, heuristics, "
                f"or default. Did the @libtuner config / heuristics chain "
                f"omit it?"
            )

    kwargs_for_cpp: List[Tuple[str, ConfigValue]] = [
        (name, _coerce_for_cpp(name, value)) for name, value in ordered.items()
    ]

    # Cross-language guard for the C++ mirror cache: how many args contribute
    # a dtype to the key (mirrors get_key's per-tensor dtype append). The C++
    # AutotunedCall appends one int64 per tensor arg and asserts this matches.
    nargs = dict(zip(autotuner.arg_names, args))
    keyed = {k: v for k, v in {**nargs, **kwargs}.items()
             if k in autotuner.arg_names}
    key_dtype_count = sum(1 for v in keyed.values() if hasattr(v, "dtype"))

    return {
        "num_warps": int(config.num_warps),
        "num_stages": int(config.num_stages),
        "kwargs": kwargs_for_cpp,
        "key_dtype_count": key_dtype_count,
        "has_pre_hook": config.pre_hook is not None,
    }


def run_pre_hook(kernel_path, function_name, args, kwargs):
    """Execute the resolved config's per-launch pre_hook (config.pre_hook),
    the ONE piece the C++ separated launch cannot do itself. Called from C++
    immediately before each autotuned_call when lookup_config reported
    has_pre_hook=True. Resolution is a cache hit here (already tuned), so the
    only real work is the hook itself. No-op if the config has no pre_hook.

    Mirrors Autotuner.run()'s pre-launch line:
        if config.pre_hook is not None:
            config.pre_hook({**nargs, **kwargs, **config.all_kwargs()})
    """
    root = _load_kernel(kernel_path, function_name)
    autotuner = _find_autotuner(root)
    config = _resolve_any(autotuner, args, kwargs)
    if config.pre_hook is None:
        return
    full_nargs = {
        **dict(zip(autotuner.arg_names, args)),
        **kwargs,
        **config.all_kwargs(),
    }
    config.pre_hook(full_nargs)
