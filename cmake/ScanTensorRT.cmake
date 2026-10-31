# S03 FP32 scan plugins (docs/SCAN_TRT.md). Included only with both
# SACCADE_ENABLE_CUDA and SACCADE_ENABLE_TENSORRT, after cmake/TensorRT.cmake
# has resolved and probed the pinned SDK. The plugins link the S02 library
# target itself, so the scan kernels keep exactly the S02 flags
# (SACCADE_SCAN_CUDA_FLAGS); the plugin and harness host code is C++.
set(SACCADE_SCAN_TRT_WARNINGS $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror>)

add_library(saccade_scan_trt src/scan_trt_plugin.cpp src/scan_trt_network.cpp)
target_include_directories(saccade_scan_trt SYSTEM PUBLIC "${SACCADE_TENSORRT_INCLUDE}")
target_link_libraries(saccade_scan_trt PUBLIC saccade_selective_scan_cuda CUDA::cudart CUDA::cuda_driver "${SACCADE_TENSORRT_LIBRARY}")

# Frame-end harness: control kernels with default CUDA flags (integer and copy
# work only) and the host session around a deserialized engine.
add_library(saccade_scan_trt_frame src/scan_trt_frame.cpp cuda/scan_frame.cu)
set_target_properties(saccade_scan_trt_frame PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(saccade_scan_trt_frame PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(saccade_scan_trt_frame PUBLIC saccade_scan_trt saccade_protocol)

add_executable(scan_trt_engine tools/scan_trt_engine.cpp)
target_link_libraries(scan_trt_engine PRIVATE saccade_scan_trt_frame saccade_file_io saccade_sha256)

add_executable(scan_trt_tests tests/scan_trt_tests.cpp tests/scan_trt_api_trace.cpp tests/scan_trt_fault.cu)
set_target_properties(scan_trt_tests PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(scan_trt_tests PRIVATE saccade_scan_trt_frame saccade_file_io CUDA::cupti)

foreach(target saccade_scan_trt saccade_scan_trt_frame scan_trt_engine scan_trt_tests)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE ${SACCADE_SCAN_TRT_WARNINGS})
  endif()
endforeach()
target_compile_options(scan_trt_tests PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
foreach(target scan_trt_engine scan_trt_tests)
  # TensorRT dlopens its per-SM builder resource by basename (see TensorRT.cmake).
  set_target_properties(${target} PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
  target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
endforeach()

add_test(NAME scan_trt COMMAND scan_trt_tests)
set_tests_properties(scan_trt PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 900)
add_test(NAME scan_trt_api_trace COMMAND scan_trt_tests --group api_trace)
set_tests_properties(scan_trt_api_trace PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 300)
# A device fault leaves the CUDA context unusable, so it runs in its own process.
add_test(NAME scan_trt_async_fault COMMAND scan_trt_tests --group async_fault)
set_tests_properties(scan_trt_async_fault PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 300)

# Serialization across processes: build and save in one, register, deserialize
# and run in another; without registration deserialization must fail.
set(SACCADE_SCAN_TRT_ENGINE "${CMAKE_CURRENT_BINARY_DIR}/scan-trt/pipeline.engine")
add_test(NAME scan_trt_engine_build COMMAND scan_trt_engine build "${SACCADE_SCAN_TRT_ENGINE}")
add_test(NAME scan_trt_engine_run COMMAND scan_trt_engine run "${SACCADE_SCAN_TRT_ENGINE}"
  "${CMAKE_CURRENT_BINARY_DIR}/scan-trt/run-report.txt")
add_test(NAME scan_trt_engine_unregistered COMMAND scan_trt_engine run-unregistered "${SACCADE_SCAN_TRT_ENGINE}")
set_tests_properties(scan_trt_engine_build PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 300 FIXTURES_SETUP scan_trt_engine)
set_tests_properties(scan_trt_engine_run scan_trt_engine_unregistered PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 300
  FIXTURES_REQUIRED scan_trt_engine)
# A fresh process must return from its first enqueue on a held stream before
# any warmup or direct S02 oracle call can load kernels on its behalf.
set_tests_properties(scan_trt_engine_run PROPERTIES ENVIRONMENT CUDA_MODULE_LOADING=LAZY)
