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

# ==============================================================================
# MLU (Cambricon MLU) Backend Configuration
# ==============================================================================

message(STATUS "Configuring MLU backend...")

if(NOT DEFINED TorchMLU_ROOT)
  execute_process(COMMAND ${Python_EXECUTABLE} "-c" "import torch_mlu;print(torch_mlu.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TorchMLU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
endif()

include(FindPackageHandleStandardArgs)

find_package(TorchMLU CONFIG REQUIRED)

if(TorchMLU_FOUND)
  message(STATUS "Found MLU Runtime.")

  if(NOT TARGET MLU::mlu_runtime)
    add_library(MLU::mlu_runtime INTERFACE IMPORTED)
    set_target_properties(MLU::mlu_runtime PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${TORCH_MLU_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${TORCH_MLU_LIBRARIES}"
    )
    message(STATUS "Created MLU::mlu_runtime imported target")
  endif()
else()
  message(FATAL_ERROR "MLU Runtime not found. Please ensure cntoolkit is installed.")
endif()

function(target_link_mlu_libraries target_name)
  target_include_directories(${target_name} INTERFACE ${TORCH_MLU_INCLUDE_DIRS})
  target_link_libraries(${target_name} INTERFACE ${TORCH_MLU_LIBRARIES})
endfunction()

message(STATUS "MLU backend configuration complete")
