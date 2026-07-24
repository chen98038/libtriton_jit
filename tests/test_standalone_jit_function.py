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

import importlib.util
import sys
import tempfile
from pathlib import Path

import triton


def load_module(path):
    spec = importlib.util.spec_from_file_location("standalone_compile_jit_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expect_exception(exception_type, fn):
    try:
        fn()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def make_token(module, source_path, function_name):
    source = source_path.read_bytes()
    return (
        f"@jit:{str(source_path.resolve()).encode().hex()}:"
        f"{function_name.encode().hex()}:{module._fnv1a64(source)}"
    )


def main():
    module = load_module(sys.argv[1])
    fixture = Path(sys.argv[2]).resolve()
    token = make_token(module, fixture, "mul_func")
    path, name, fingerprint = module._parse_jitfunction_token(token)

    assert path == str(fixture)
    assert name == "mul_func"
    assert fingerprint == module._fnv1a64(fixture.read_bytes())
    assert isinstance(
        module._load_jitfunction(path, name, fingerprint),
        triton.runtime.JITFunction,
    )
    resolved = module._resolve_jitfunction_constants([token, "*fp32"])
    assert list(resolved) == [0]
    assert isinstance(resolved[0], triton.runtime.JITFunction)
    assert module.generate_arg_layout([token, "*fp32", "i32"], []) == [
        {"type": "ptr", "dtype": "fp32"},
        {"type": "i32"},
    ]

    invalid_tokens = [
        "jit:00:00:0000000000000000",
        "@jit:00:00",
        "@jit:0:61:0000000000000000",
        "@jit:zz:61:0000000000000000",
        "@jit::61:0000000000000000",
        "@jit:2f746d70::0000000000000000",
        "@jit:2f746d70:61:1234",
        "@jit:2f746d70:61:gggggggggggggggg",
    ]
    for invalid in invalid_tokens:
        expect_exception(ValueError, lambda value=invalid: module._parse_jitfunction_token(value))

    expect_exception(
        ValueError,
        lambda: module._load_jitfunction(path, name, "0000000000000000"),
    )
    expect_exception(
        AttributeError,
        lambda: module._load_jitfunction(path, "missing_function", fingerprint),
    )
    expect_exception(
        TypeError,
        lambda: module._load_jitfunction(path, "not_a_jit_function", fingerprint),
    )

    with tempfile.TemporaryDirectory() as directory:
        mutable_path = Path(directory) / "mutable.py"
        first_source = (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def operation(x, y):\n"
            "    return x * y\n"
        )
        second_source = first_source.replace("x * y", "x + y")
        assert len(first_source) == len(second_source)
        mutable_path.write_text(first_source, encoding="utf-8")
        first_fingerprint = module._fnv1a64(mutable_path.read_bytes())
        first_function = module._load_jitfunction(
            str(mutable_path), "operation", first_fingerprint
        )
        mutable_path.write_text(second_source, encoding="utf-8")
        second_fingerprint = module._fnv1a64(mutable_path.read_bytes())
        second_function = module._load_jitfunction(
            str(mutable_path), "operation", second_fingerprint
        )
        assert first_fingerprint != second_fingerprint
        assert first_function.src != second_function.src

        cycle_path = Path(directory) / "cycle.py"
        cycle_path.write_text(
            "class Wrapper:\n"
            "    pass\n"
            "cycle = Wrapper()\n"
            "cycle.fn = cycle\n",
            encoding="utf-8",
        )
        cycle_fingerprint = module._fnv1a64(cycle_path.read_bytes())
        expect_exception(
            TypeError,
            lambda: module._load_jitfunction(
                str(cycle_path), "cycle", cycle_fingerprint
            ),
        )


if __name__ == "__main__":
    main()
