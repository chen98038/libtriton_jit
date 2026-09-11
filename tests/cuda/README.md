# Optional autotune CUDA checks

The default CTest suite covers configuration tables, export, resolver conversion,
freeze and CPU tensor/concurrency regressions. CUDA integration checks are off
by default and require a CUDA build, the Torch resolver, Triton and an available
GPU. They do not require FlagGems.

Set `REPO_ROOT` and `BUILD_DIR` to the checkout and its configured build directory.
Select an available GPU with `CUDA_VISIBLE_DEVICES`, then run:

```sh
cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DBUILD_TESTING=ON -DTRITON_JIT_BUILD_TORCH_RESOLVER=ON \
  -DTRITON_JIT_TEST_CUDA=ON
cmake --build "$BUILD_DIR" --parallel
TRITON_CACHE_DIR="$(mktemp -d)" \
  ctest --test-dir "$BUILD_DIR" -L cuda --output-on-failure
```

The three tests check source identity through the Python bridge, GIL/future
waiting, and prepare/freeze/graph capture. The graph test verifies preparation
leaves output unchanged, frozen/captured misses reject cold work, and five
replays with changed inputs add no compiler/resolver calls. CTest limits each
test to 120 seconds.

## Memory checks

Run these separately when changing benchmark argument handling. They require
`compute-sanitizer`; each layout runs in a separate process and cache:

```sh
for layout in contiguous transpose slice offset alias kwargs reset restore; do
  TRITON_CACHE_DIR="$(mktemp -d)" \
    compute-sanitizer --tool memcheck --error-exitcode 99 \
    python "$REPO_ROOT/tests/cuda/layout_safety.py" "$REPO_ROOT/scripts" "$layout" || exit $?
done
```
