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
"""Tests for scripts/tuned_resolver.py with stand-in tuner objects: the
resolver must reuse the tuner's own key, caches, pruning and policy, store the
best config under the normalised key, and describe it the way TunedTable
expects. No FlagGems, Triton or device needed.

Usage: python tests/test_tuned_resolver.py <path to scripts dir>
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from itertools import starmap
from typing import Any, Dict, List

failures = 0


def check(condition: bool, message: str) -> None:
    global failures
    if not condition:
        failures += 1
        print(f"FAILED: {message}", file=sys.stderr)


def load(scripts_dir: str, name: str):
    path = os.path.join(scripts_dir, name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---- stand-ins --------------------------------------------------------------


class Config:
    def __init__(
        self, kwargs, num_warps=4, num_stages=3, num_ctas=1, maxnreg=None, pre_hook=None
    ):
        self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages
        self.num_ctas, self.maxnreg, self.pre_hook = num_ctas, maxnreg, pre_hook

    def all_kwargs(self):
        return {
            **self.kwargs,
            "num_warps": self.num_warps,
            "num_ctas": self.num_ctas,
            "num_stages": self.num_stages,
            "maxnreg": self.maxnreg,
        }

    def __repr__(self):
        return f"Config({self.kwargs}, nw={self.num_warps})"


class JITFunction:
    def __init__(self, fn, arg_names):
        self.fn, self.arg_names, self.cache_key = fn, arg_names, "ck"


class Tensor:
    def __init__(self, dtype):
        self.dtype = dtype


class KVCache:
    def __init__(self):
        self.rows: Dict[Any, Any] = {}

    def get(self, key):
        return self.rows.get(key)

    def __getitem__(self, key):
        return self.rows[key]

    def __setitem__(self, key, value):
        self.rows[key] = value

    def __contains__(self, key):
        return key in self.rows


class LibCache:
    def __init__(self):
        self.tables: Dict[Any, KVCache] = {}

    def __getitem__(self, key):
        return self.tables.setdefault(key, KVCache())


def default_strategy(key):
    return key


def log2_strategy(key):
    return 2 ** math.ceil(math.log2(key))


def align32_strategy(key):
    if key == 0:
        return 0
    if key < 32:
        return 2 ** math.ceil(math.log2(key))
    return math.ceil(key / 32) * 32


class LibTuner:
    _strategy_table = {
        None: default_strategy,
        "default": default_strategy,
        "log": log2_strategy,
        "align32": align32_strategy,
    }

    def __init__(self, name, fn, keys, strategy, configs, timings):
        self.__name__, self.fn, self.keys, self.strategy, self.configs = (
            name,
            fn,
            keys,
            strategy,
            configs,
        )
        self.arg_names = fn.arg_names
        self.configs_hash = "h" * 32
        self.config_table_name = f"{name}_hash"
        self.benchmark_table_name = f"{name}_bench"
        self.cache = KVCache()
        self.timings = timings  # config index -> p50
        self.trace = (
            {}
        )  # Shared observer; invocation state lives on a local tuner copy.
        self.bench_calls: List[Any] = []
        self.pre_hook_calls = 0
        self.policy_calls = 0
        self.nargs = None

    @property
    def policy_calls(self):
        return self.trace.get("policy", 0)

    @policy_calls.setter
    def policy_calls(self, value):
        self.trace["policy"] = value

    @property
    def pre_hook_calls(self):
        return self.trace.get("hook", 0)

    @pre_hook_calls.setter
    def pre_hook_calls(self, value):
        self.trace["hook"] = value

    @property
    def last_bench_args(self):
        return self.trace["args"]

    @last_bench_args.setter
    def last_bench_args(self, value):
        self.trace["args"] = value

    def get_key(self, args):
        if self.strategy is None:
            key = tuple(args[k] for k in self.keys if k in args)
        else:
            key = tuple(
                starmap(lambda i, k: self.strategy[i](args[k]), enumerate(self.keys))
            )
        key += tuple(str(v.dtype) for v in args.values() if isinstance(v, Tensor))
        return key

    def get_benchmark_key(self, args):
        return tuple(args[k] for k in self.keys if k in args) + ("proto",)

    def prune_configs(self, kwargs):
        return [
            c
            for c in self.configs
            if c.kwargs.get("BLOCK_M", 0) <= kwargs.get("max_block", 1 << 30)
        ]

    def _bench(self, *args, config, **kwargs):
        self.bench_calls.append(config)
        assert callable(kwargs.get("grid")) or isinstance(
            kwargs.get("grid"), tuple
        ), "grid must reach _bench"
        assert kwargs.get("warmup") is False, "warmup=False must reach _bench"
        self.last_bench_args = args
        p50 = self.timings[self.configs.index(config)]
        return (p50, p50 * 0.9, p50 * 1.1)

    def policy(self, bench_fn, configs, args, kwargs):
        self.policy_calls += 1
        timings = {c: bench_fn(c)[0] for c in configs}
        best = min(timings, key=timings.get)
        return best, timings

    def pre_hook(self, full_nargs, reset_only=False):
        self.pre_hook_calls += 1


def fake_libentry(package):
    module = types.ModuleType(f"{package}.utils.libentry")
    module.LibTuner = LibTuner
    module.libcache = LibCache()
    return module


def mm_kernel(a, b, c, M, N, K, BLOCK_M, BLOCK_N, EVEN_K):  # noqa: N803
    pass


def run(scripts_dir: str) -> int:
    load(scripts_dir, "export_tuned_table")
    resolver = load(scripts_dir, "tuned_resolver")
    libentry = fake_libentry("flag_gems")
    configs = [
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "EVEN_K": True}, num_warps=8, num_ctas=2
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "EVEN_K": False}, num_warps=4, maxnreg=128
        ),
        Config({"BLOCK_M": 256, "BLOCK_N": 128, "EVEN_K": True}, num_warps=8),
    ]
    tuner = LibTuner(
        "mm",
        JITFunction(
            mm_kernel, ["a", "b", "c", "M", "N", "K", "BLOCK_M", "BLOCK_N", "EVEN_K"]
        ),
        ["M", "N", "K"],
        [log2_strategy, log2_strategy, align32_strategy],
        configs,
        timings=[5.0, 3.0, 9.0],
    )
    a, b, c = Tensor("torch.float16"), Tensor("torch.float16"), Tensor("torch.float32")
    args = (a, b, c, 1000, 4000, 4090)
    kwargs = {"max_block": 200}

    # an untuned key without a grid is reported, not benchmarked
    nogrid = resolver.resolve_with_tuner(tuner, libentry, args, kwargs)
    check(
        "supplied no launch grid" in nogrid.get("unsupported", ""),
        f"no-grid answer: {nogrid}",
    )
    check(tuner.policy_calls == 0, "no benchmark without a grid")
    grid_expr = "(math.ceil(M / META['BLOCK_M']) * math.ceil(N / META['BLOCK_N']),)"
    out = resolver.resolve_with_tuner(tuner, libentry, args, kwargs, grid=grid_expr)
    check("unsupported" not in out, f"resolved: {out}")
    check(
        out.get("source_path", "").endswith("test_tuned_resolver.py"),
        "source_path reported",
    )
    check(out["kernel_id"] == "mm_kernel" and out["op_name"] == "mm", "identity")
    check(
        out["key"]
        == [1024, 4096, 4096, "torch.float16", "torch.float16", "torch.float32"],
        f"normalised key {out['key']}",
    )
    check(
        [c["strategy"] for c in out["key_columns"]] == ["log", "log", "align32"],
        "strategies",
    )
    check(out["dtype_keys"] == 3, "dtype keys")
    check(
        out["kwargs"] == [["BLOCK_M", 64], ["BLOCK_N", 64], ["EVEN_K", False]],
        f"kwargs {out['kwargs']}",
    )
    check(out["num_warps"] == 4 and out["num_stages"] == 3, "nw/ns")
    check(out["extra"] == {"maxnreg": "128"}, f"extra {out['extra']}")
    # pruning happened through the tuner (BLOCK_M 256 excluded), policy picked the fastest, key stored normalised
    check(
        tuner.policy_calls == 1 and len(tuner.bench_calls) == 2,
        f"policy/bench calls {tuner.policy_calls}/{len(tuner.bench_calls)}",
    )
    check(
        (1024, 4096, 4096, "torch.float16", "torch.float16", "torch.float32")
        in tuner.cache,
        "stored under get_key()",
    )
    check(
        tuner.pre_hook_calls == 1 and tuner.nargs is None,
        "pre_hook reset called, nargs cleared",
    )
    grid_fn = resolver.build_grid(grid_expr, tuner, args, {})
    check(
        grid_fn({"BLOCK_M": 64, "BLOCK_N": 64}) == (16 * 63,),
        f"grid expression evaluated: {grid_fn({'BLOCK_M': 64, 'BLOCK_N': 64})}",
    )
    check(
        resolver.build_grid((4, 1), tuner, args, {}) == (4, 1), "tuple grid passthrough"
    )
    # second call with a shape in the same buckets: cache hit, no benchmark
    out2 = resolver.resolve_with_tuner(
        tuner, libentry, (a, b, c, 1024, 4096, 4096), kwargs
    )
    check(
        out2["kwargs"] == out["kwargs"] and tuner.policy_calls == 1,
        "cache hit skips policy",
    )
    # benchmark cache is reused for the same raw key even when the config cache is bypassed
    bench_table = libentry.libcache[
        tuner.benchmark_table_name, (1000, 4000, 4090, "proto")
    ]
    check(len(bench_table.rows) == 2, f"benchmark cache rows {len(bench_table.rows)}")
    # different dtype -> different key -> benchmark again
    out3 = resolver.resolve_with_tuner(
        tuner,
        libentry,
        (Tensor("torch.bfloat16"), b, c, 1000, 4000, 4090),
        kwargs,
        grid=grid_expr,
    )
    check(
        out3["key"][3] == "torch.bfloat16" and tuner.policy_calls == 2,
        "dtype changes the key",
    )
    # a config with pre_hook is refused
    hooked = LibTuner(
        "mm_hooked",
        tuner.fn,
        ["M"],
        None,
        [
            Config(
                {"BLOCK_M": 8, "BLOCK_N": 8, "EVEN_K": True}, pre_hook=lambda n: None
            ),
            Config({"BLOCK_M": 16, "BLOCK_N": 8, "EVEN_K": True}),
        ],
        timings=[1.0, 2.0],
    )
    out4 = resolver.resolve_with_tuner(hooked, libentry, args, {}, grid=(1,))
    check("pre_hook" in out4.get("unsupported", ""), f"pre_hook refused: {out4}")
    # single-config tuner: no benchmarking, config[0]
    single = LibTuner(
        "mm_single",
        tuner.fn,
        ["M"],
        None,
        [Config({"BLOCK_M": 32, "BLOCK_N": 32, "EVEN_K": True})],
        timings=[1.0],
    )
    out5 = resolver.resolve_with_tuner(single, libentry, args, {})
    check(
        out5["kwargs"][0] == ["BLOCK_M", 32] and single.policy_calls == 0,
        "single config short-circuits",
    )
    # a kwarg that is not a kernel parameter is refused rather than silently dropped
    odd = LibTuner(
        "mm_odd",
        tuner.fn,
        ["M"],
        None,
        [
            Config({"BLOCK_M": 32, "BLOCK_N": 32, "EVEN_K": True, "GHOST": 1}),
            Config({"BLOCK_M": 64, "BLOCK_N": 32, "EVEN_K": True, "GHOST": 2}),
        ],
        timings=[2.0, 1.0],
    )
    out6 = resolver.resolve_with_tuner(odd, libentry, args, {}, grid=(1,))
    check("GHOST" in out6.get("unsupported", ""), f"ghost kwarg refused: {out6}")
    # ---- find_tuner: same kernel name in two files, and the same file loaded twice ----
    import gc

    export = sys.modules["export_tuned_table"]
    pkg = types.ModuleType("flag_fake")
    pkg.__path__ = []
    utils = types.ModuleType("flag_fake.utils")
    utils.__path__ = []
    sys.modules["flag_fake"] = pkg
    sys.modules["flag_fake.utils"] = utils
    sys.modules["flag_fake.utils.libentry"] = libentry
    this_file = os.path.abspath(__file__)

    def kernel_in(module_name):
        def dup_kernel(a, M, BLOCK):  # noqa: N803
            pass

        dup_kernel.__module__ = module_name
        return dup_kernel

    # a registered package module whose __file__ is this test file
    ops_mod = types.ModuleType("flag_fake.ops.dup")
    ops_mod.__file__ = this_file
    sys.modules["flag_fake.ops.dup"] = ops_mod
    registered_fn = kernel_in("flag_fake.ops.dup")
    stray_fn = kernel_in("dup")  # what a second spec_from_file_location load looks like
    registered = LibTuner(
        "dup",
        JITFunction(registered_fn, ["a", "M", "BLOCK"]),
        ["M"],
        None,
        [Config({"BLOCK": 1}), Config({"BLOCK": 2})],
        [1.0, 2.0],
    )
    registered.base_fn = registered_fn
    stray = LibTuner(
        "dup",
        JITFunction(stray_fn, ["a", "M", "BLOCK"]),
        ["M"],
        None,
        [Config({"BLOCK": 1}), Config({"BLOCK": 2})],
        [1.0, 2.0],
    )
    stray.base_fn = stray_fn
    gc.collect()
    found = resolver.find_tuner("flag_fake", "dup_kernel", source_path=this_file)
    check(found is registered, "find_tuner prefers the copy imported under the package")
    check(
        resolver.find_tuner("flag_fake", "dup_kernel") is registered,
        "without source_path the same single-file case still resolves",
    )
    check(
        export.dedupe_tuners([stray, registered], "flag_fake") == [registered],
        "exporter dedupes copies of one file",
    )
    try:
        resolver.find_tuner("flag_fake", "no_such_kernel")
        check(False, "unknown kernel must raise")
    except LookupError:
        pass
    if failures:
        print(f"{failures} check(s) failed", file=sys.stderr)
        return 1
    print("tuned_resolver: all checks passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(run(sys.argv[1]))
