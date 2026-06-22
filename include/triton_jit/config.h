#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace triton_jit {

using ConfigValue = std::variant<int64_t, bool>;

struct Config {
  int num_warps;
  int num_stages;
  std::vector<std::pair<std::string, ConfigValue>> kwargs;
};

}  // namespace triton_jit
