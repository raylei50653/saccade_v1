option(SACCADE_ENABLE_TENSORRT "Build the native TensorRT SDK smoke test (requires a runnable GPU)" OFF)
set(SACCADE_TENSORRT_ROOT "" CACHE PATH "TensorRT 10.16.1.11 CUDA 13.x native SDK root")
if(NOT SACCADE_ENABLE_TENSORRT)
  return()
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  message(FATAL_ERROR "P01-A native TensorRT target currently supports Linux x86-64 only")
endif()
if(NOT IS_DIRECTORY "${SACCADE_TENSORRT_ROOT}")
  message(FATAL_ERROR "Set SACCADE_TENSORRT_ROOT to the fresh TensorRT 10.16.1.11 CUDA 13.x SDK root")
endif()
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 13.2 OR NOT CUDAToolkit_VERSION VERSION_LESS 14)
  message(FATAL_ERROR "TensorRT 10.16.1.11 CUDA 13.2 package requires a CUDA Toolkit in [13.2,14); found ${CUDAToolkit_VERSION}")
endif()
# NO_DEFAULT_PATH prevents silently using an unrelated system or Python SDK.
# Resolve again on reconfigure: a changed SDK root must not reuse cached paths.
unset(SACCADE_TENSORRT_INCLUDE CACHE)
unset(SACCADE_TENSORRT_LIBRARY CACHE)
find_path(SACCADE_TENSORRT_INCLUDE NvInfer.h
  PATHS "${SACCADE_TENSORRT_ROOT}/include" "${SACCADE_TENSORRT_ROOT}/usr/include/x86_64-linux-gnu"
  NO_DEFAULT_PATH REQUIRED)
find_file(SACCADE_TENSORRT_LIBRARY NAMES libnvinfer.so.10.16.1 libnvinfer.so.10
  PATHS "${SACCADE_TENSORRT_ROOT}/lib" "${SACCADE_TENSORRT_ROOT}/lib64"
        "${SACCADE_TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
get_filename_component(SACCADE_TENSORRT_LIBRARY "${SACCADE_TENSORRT_LIBRARY}" REALPATH)
get_filename_component(SACCADE_TENSORRT_LIBDIR "${SACCADE_TENSORRT_LIBRARY}" DIRECTORY)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/trt-sdk-probe.cpp" [=[
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
static_assert(NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 16 && NV_TENSORRT_PATCH == 1 && NV_TENSORRT_BUILD == 11,
              "native TensorRT headers must be 10.16.1.11");
int main() {
  int runtime = 0, driver = 0;
  cudaDeviceProp gpu{};
  if (getInferLibVersion() != NV_TENSORRT_VERSION || getInferLibBuildVersion() != NV_TENSORRT_BUILD) {
    std::cerr << "TensorRT header/library version mismatch (requires 10.16.1.11)\n"; return 1;
  }
  if (cudaRuntimeGetVersion(&runtime) != cudaSuccess || runtime < 13020 || runtime >= 14000 ||
      cudaDriverGetVersion(&driver) != cudaSuccess || driver < runtime ||
      cudaGetDeviceProperties(&gpu, 0) != cudaSuccess || gpu.major < 7 || (gpu.major == 7 && gpu.minor < 5)) {
    std::cerr << "CUDA runtime/driver/GPU unavailable or incompatible: runtime=" << runtime << " driver=" << driver << '\n'; return 2;
  }
  std::cout << "TensorRT=" << getInferLibVersion() << " CUDA runtime=" << runtime << " driver=" << driver
            << " GPU=" << gpu.name << " SM=" << gpu.major << '.' << gpu.minor << '\n';
}
]=])
try_run(SACCADE_TRT_PROBE_EXIT SACCADE_TRT_PROBE_COMPILED
  "${CMAKE_CURRENT_BINARY_DIR}/trt-sdk-probe" "${CMAKE_CURRENT_BINARY_DIR}/trt-sdk-probe.cpp"
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${SACCADE_TENSORRT_INCLUDE};${CUDAToolkit_INCLUDE_DIRS}"
  LINK_LIBRARIES "${SACCADE_TENSORRT_LIBRARY}" CUDA::cudart
  COMPILE_OUTPUT_VARIABLE SACCADE_TRT_PROBE_BUILD RUN_OUTPUT_VARIABLE SACCADE_TRT_PROBE_OUTPUT)
if(NOT SACCADE_TRT_PROBE_COMPILED OR NOT SACCADE_TRT_PROBE_EXIT EQUAL 0)
  message(FATAL_ERROR "Native TensorRT SDK probe failed; no fallback.\n${SACCADE_TRT_PROBE_BUILD}\n${SACCADE_TRT_PROBE_OUTPUT}")
endif()
message(STATUS "Native SDK probe: ${SACCADE_TRT_PROBE_OUTPUT}")
if(NOT SACCADE_TRT_PROBE_OUTPUT MATCHES "SM=([0-9]+)\\.([0-9]+)")
  message(FATAL_ERROR "Cannot identify GPU SM from native SDK probe")
endif()
set(SACCADE_TRT_RESOURCE "${SACCADE_TENSORRT_LIBDIR}/libnvinfer_builder_resource_sm${CMAKE_MATCH_1}${CMAKE_MATCH_2}.so.10.16.1")
if(NOT EXISTS "${SACCADE_TRT_RESOURCE}")
  message(FATAL_ERROR "Pinned TensorRT builder resource missing for this GPU: ${SACCADE_TRT_RESOURCE}")
endif()
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE SACCADE_TRT_BUILD_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND git status --porcelain WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  OUTPUT_VARIABLE SACCADE_TRT_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(SACCADE_TRT_GIT_STATUS STREQUAL "")
  set(SACCADE_TRT_BUILD_DIRTY false)
else()
  set(SACCADE_TRT_BUILD_DIRTY true)
endif()
add_executable(tensorrt_smoke tools/tensorrt_smoke.cpp)
target_include_directories(tensorrt_smoke SYSTEM PRIVATE "${SACCADE_TENSORRT_INCLUDE}")
target_link_libraries(tensorrt_smoke PRIVATE saccade_file_io saccade_sha256 CUDA::cudart "${SACCADE_TENSORRT_LIBRARY}" ${CMAKE_DL_LIBS})
target_compile_definitions(tensorrt_smoke PRIVATE
  SACCADE_TRT_BUILD_COMMIT="${SACCADE_TRT_BUILD_COMMIT}" SACCADE_TRT_BUILD_DIRTY="${SACCADE_TRT_BUILD_DIRTY}"
  SACCADE_TRT_HEADER_PATH="${SACCADE_TENSORRT_INCLUDE}/NvInferVersion.h"
  SACCADE_TRT_EXPECTED_LIBRARY="${SACCADE_TENSORRT_LIBRARY}"
  SACCADE_TRT_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}")
set_target_properties(tensorrt_smoke PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
# TensorRT dlopens its per-SM builder resource by basename. ELF RUNPATH is not
# transitive to that dependency, so use RPATH for this isolated SDK executable.
target_link_options(tensorrt_smoke PRIVATE "LINKER:--disable-new-dtags")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tensorrt_smoke PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME native_tensorrt_roundtrip COMMAND tensorrt_smoke
  "${CMAKE_CURRENT_BINARY_DIR}/native-smoke.engine" "${CMAKE_CURRENT_BINARY_DIR}/native-smoke.txt")
set_tests_properties(native_tensorrt_roundtrip PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 180)
add_test(NAME native_tensorrt_cli_io COMMAND "${CMAKE_COMMAND}"
  -DPROGRAM=$<TARGET_FILE:tensorrt_smoke>
  -DFIXTURE_DIR=${CMAKE_CURRENT_BINARY_DIR}/native-io-fixtures
  -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TensorRTIOTest.cmake)
set_tests_properties(native_tensorrt_cli_io PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 180)
