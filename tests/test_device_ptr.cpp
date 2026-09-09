// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Signature and runtime-ABI behavior of TritonDevicePtr, checked against the
// production ArgHandle. The static signature is built by hand so the test needs
// neither a device nor the Python bridge.

#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "triton_jit/triton_jit_function.h"

namespace {

int g_failures = 0;

bool expect(bool condition, const char* message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
    ++g_failures;
  }
  return condition;
}

struct Emitted {
  std::vector<std::string> signature;
  size_t abi_args;
  int consumed;
};

// Run one argument list through ArgHandle with the given per-argument kinds.
template <typename... Args>
Emitted emit(const std::vector<triton_jit::ArgType>& kinds, Args... args) {
  using namespace triton_jit;
  StaticSignature static_signature {static_cast<int>(kinds.size()), kinds};
  ParameterBuffer buffer;
  c10::SmallVector<std::string> signature;
  ArgHandle handler {static_signature, buffer, signature, 0};
  handler.handle_args(args...);
  return Emitted {std::vector<std::string>(signature.begin(), signature.end()), buffer.size(), handler.idx};
}

void expect_token(const Emitted& e, size_t index, const char* expected, const char* what) {
  if (!expect(e.signature.size() > index, what)) {
    return;
  }
  if (e.signature[index] != expected) {
    std::cerr << "FAIL: " << what << " (got \"" << e.signature[index] << "\", expected \"" << expected
              << "\")\n";
    ++g_failures;
  }
}

// Addresses are never dereferenced; only their alignment matters here.
constexpr std::uintptr_t kAligned = 0x7F0000001000ULL;     // 16-byte aligned
constexpr std::uintptr_t kMisaligned = 0x7F0000001004ULL;  // 4 bytes off
constexpr std::uintptr_t kAddressOne = 1ULL;               // must never fold to ":1"

struct Interleaved {
  float re;
  float im;
};

struct StorageHalf {
  std::uint16_t bits;
};

}  // namespace

// A downstream library maps its own storage type onto a Triton dtype.
template <>
struct triton_jit::triton_dtype_of<StorageHalf> {
  static constexpr triton_jit::TritonDType value = triton_jit::TritonDType::kFp16;
};

int main() {
  using namespace triton_jit;

  // ---- dtype tags ---------------------------------------------------------
  static_assert(std::is_trivially_copyable_v<TritonDevicePtr>);
  static_assert(triton_dtype_of<float>::value == TritonDType::kFp32);
  static_assert(triton_dtype_of<double>::value == TritonDType::kFp64);
  static_assert(triton_dtype_of<bool>::value == TritonDType::kI1);
  static_assert(triton_dtype_of<std::int32_t>::value == TritonDType::kI32);
  static_assert(triton_dtype_of<std::uint64_t>::value == TritonDType::kU64);
  // long and long long are distinct types of the same width; both must map.
  static_assert(triton_dtype_of<long>::value == triton_dtype_of<long long>::value);
  expect(std::string(to_triton_typename(TritonDType::kFp8E4NV)) == "fp8e4nv",
         "fp8e4nv must round-trip through to_triton_typename");
  expect(std::string(to_triton_typename(TritonDType::kBf16)) == "bf16",
         "bf16 must round-trip through to_triton_typename");

  // ---- factories ----------------------------------------------------------
  {
    float storage = 0.0F;
    const double const_storage = 0.0;
    Interleaved complex_storage {};
    StorageHalf half_storage {};

    expect(device_ptr(&storage).dtype == TritonDType::kFp32,
           "pointee-deduced factory must map float* to fp32");
    expect(device_ptr(&const_storage).dtype == TritonDType::kFp64,
           "pointee-deduced factory must see through const");
    expect(device_ptr<float>(&complex_storage).dtype == TritonDType::kFp32,
           "explicit-element factory must reinterpret an interleaved buffer as fp32");
    expect(device_ptr<float>(&storage).dtype == TritonDType::kFp32,
           "explicit-element factory must also accept a matching pointee type");
    expect(device_ptr(&half_storage).dtype == TritonDType::kFp16,
           "a downstream triton_dtype_of specialization must be honored");

    const TritonDevicePtr handle = device_ptr<float>(kAligned);
    expect(handle.value == kAligned && handle.dtype == TritonDType::kFp32,
           "integer-handle factory must preserve the address and element type");
    const TritonDevicePtr runtime = device_ptr(&storage, TritonDType::kFp8E5);
    expect(runtime.dtype == TritonDType::kFp8E5,
           "runtime-dtype factory must accept a dtype chosen at run time");
  }

  // ---- pointer specialization --------------------------------------------
  const std::vector<ArgType> one_specialized {ArgType::SPECIALIZED};
#if defined(BACKEND_NPU)
  // NPU keeps every argument in the ABI and emits no specialization hints, so
  // the arg list stays consistent with the layout the launcher parses.
  const char* const kAlignedPtr = "*fp32";
  const char* const kAlignedI32 = "i32";
  const char* const kEqualToOne = "i32";
  const size_t kEqualToOneAbi = 1;
#else
  const char* const kAlignedPtr = "*fp32:16";
  const char* const kAlignedI32 = "i32:16";
  const char* const kEqualToOne = "i32:1";
  const size_t kEqualToOneAbi = 0;
#endif

  {
    const Emitted aligned = emit(one_specialized, device_ptr<float>(kAligned));
    expect_token(aligned, 0, kAlignedPtr, "an aligned device pointer must carry the ':16' hint");
    expect(aligned.abi_args == 1, "a device pointer always stays in the runtime ABI");
    expect(aligned.consumed == 1, "a device pointer must consume one signature slot");

    const Emitted misaligned = emit(one_specialized, device_ptr<float>(kMisaligned));
    expect_token(misaligned, 0, "*fp32", "a misaligned device pointer must not claim ':16'");
    expect(misaligned.abi_args == 1, "a misaligned device pointer stays in the runtime ABI");

    // An address equal to 1 must not be mistaken for an equal-to-one integer:
    // that would fold the pointer away and drop it from the runtime ABI.
    const Emitted address_one = emit(one_specialized, device_ptr<float>(kAddressOne));
    expect_token(address_one, 0, "*fp32", "a device pointer must never be specialized to ':1'");
    expect(address_one.abi_args == 1, "a device pointer must never leave the runtime ABI");

    const Emitted null_ptr = emit(one_specialized, device_ptr<float>(std::uintptr_t {0}));
    expect_token(null_ptr, 0, kAlignedPtr, "a null device pointer is trivially aligned");
  }

  // ---- dtypes reach the signature ----------------------------------------
  {
    const Emitted half = emit(one_specialized, device_ptr(kAligned, TritonDType::kFp16));
#if defined(BACKEND_NPU)
    expect_token(half, 0, "*fp16", "a runtime dtype must reach the signature");
#else
    expect_token(half, 0, "*fp16:16", "a runtime dtype must reach the signature");
#endif
    const Emitted fp8 = emit(one_specialized, device_ptr(kAligned, TritonDType::kFp8E4NV));
#if defined(BACKEND_NPU)
    expect_token(fp8, 0, "*fp8e4nv", "fp8 has no C++ type and must come from the runtime dtype");
#else
    expect_token(fp8, 0, "*fp8e4nv:16", "fp8 has no C++ type and must come from the runtime dtype");
#endif
  }

  // ---- other ArgType kinds ------------------------------------------------
  {
    const Emitted plain = emit({ArgType::NON_CONSTEXPR}, device_ptr<float>(kAligned));
    expect_token(plain, 0, "*fp32", "a non-specialized device pointer must not carry an alignment hint");
    expect(plain.abi_args == 1, "a non-specialized device pointer stays in the runtime ABI");

    const Emitted no_alignment = emit({ArgType::SPECIALIZED_NO_ALIGNMENT}, device_ptr<float>(kAligned));
    expect_token(no_alignment, 0, "*fp32", "do_not_specialize_on_alignment must suppress the ':16' hint");
    expect(no_alignment.abi_args == 1, "do_not_specialize_on_alignment keeps the pointer in the runtime ABI");
  }

  // ---- optional device pointers ------------------------------------------
  {
    const Emitted present =
        emit(one_specialized, std::optional<TritonDevicePtr> {device_ptr<float>(kAligned)});
    expect_token(present, 0, kAlignedPtr, "an engaged optional behaves like a plain pointer");
    expect(present.abi_args == 1, "an engaged optional stays in the runtime ABI");

    const Emitted absent = emit(one_specialized, std::optional<TritonDevicePtr> {});
    expect_token(absent, 0, "nullopt", "a disengaged optional must become a constexpr None");
    expect(absent.abi_args == 0, "a disengaged optional must not enter the runtime ABI");
  }

  // ---- a full argument list, mixing pointers with the existing kinds ------
  {
    // (a, x, y, alpha, m, incx, BLOCK) -- the shape of a C BLAS level-2 launch.
    const std::vector<ArgType> kinds {
        ArgType::SPECIALIZED,    // a
        ArgType::SPECIALIZED,    // x
        ArgType::SPECIALIZED,    // y
        ArgType::NON_CONSTEXPR,  // alpha
        ArgType::SPECIALIZED,    // m
        ArgType::SPECIALIZED,    // incx
        ArgType::CONSTEXPR,      // BLOCK
    };
    const Emitted e = emit(kinds,
                           device_ptr<float>(kAligned),
                           device_ptr<float>(kAligned),
                           device_ptr<float>(kMisaligned),
                           1.0F,
                           static_cast<int64_t>(4096),
                           static_cast<int64_t>(1),
                           32);
    expect(e.consumed == 7, "every argument must consume exactly one signature slot");
    expect_token(e, 0, kAlignedPtr, "aligned matrix pointer");
    expect_token(e, 2, "*fp32", "misaligned output pointer");
    expect_token(e, 3, "fp32", "a float scalar is unaffected by pointer handling");
    expect_token(e, 4, kAlignedI32, "a divisible size keeps its ':16' hint");
    expect_token(e, 5, kEqualToOne, "an equal-to-one stride keeps its existing behavior");
    expect_token(e, 6, "32", "a constexpr still emits its value");
    // pointers(3) + alpha(1) + m(1) + incx(0 or 1); the constexpr never joins.
    expect(e.abi_args == 5 + kEqualToOneAbi, "runtime ABI must hold exactly the non-folded arguments");
  }

  if (g_failures != 0) {
    std::cerr << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << "test_device_ptr: all checks passed\n";
  return 0;
}
