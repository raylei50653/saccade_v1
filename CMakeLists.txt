cmake_minimum_required(VERSION 3.20)
project(saccade_v1_protocol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(saccade_protocol src/protocol.cpp)
target_include_directories(saccade_protocol PUBLIC include)

add_executable(protocol_model tools/protocol_model.cpp)
add_executable(packet_compare tools/packet_compare.cpp)
target_link_libraries(packet_compare PRIVATE saccade_protocol)
add_executable(protocol_tests tests/protocol_tests.cpp)
target_link_libraries(protocol_tests PRIVATE saccade_protocol)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_protocol protocol_model packet_compare protocol_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endforeach()
endif()


enable_testing()
add_test(NAME protocol_happens_before COMMAND protocol_model --self-test)
add_test(NAME protocol_state_and_packet COMMAND protocol_tests)
add_test(NAME packet_compare_cli
  COMMAND ${CMAKE_COMMAND}
    -DPROTOCOL_TESTS=$<TARGET_FILE:protocol_tests>
    -DPACKET_COMPARE=$<TARGET_FILE:packet_compare>
    -DFIXTURE_DIR=${CMAKE_CURRENT_BINARY_DIR}/packet-fixtures
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PacketCompareTest.cmake)

add_library(saccade_sha256 src/sha256.cpp)
target_include_directories(saccade_sha256 PUBLIC include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(saccade_sha256 PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_library(saccade_synthetic_config src/synthetic_config.cpp)
target_include_directories(saccade_synthetic_config PUBLIC include)
target_link_libraries(saccade_synthetic_config PUBLIC saccade_sha256)
add_executable(synthetic_config_tests tests/synthetic_config_tests.cpp)
target_link_libraries(synthetic_config_tests PRIVATE saccade_synthetic_config)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(saccade_synthetic_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(synthetic_config_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME synthetic_resolved_config COMMAND synthetic_config_tests)

add_library(saccade_file_io src/file_io.cpp)
target_include_directories(saccade_file_io PUBLIC include)
add_executable(file_io_tests tests/file_io_tests.cpp)
target_link_libraries(file_io_tests PRIVATE saccade_file_io)
add_executable(cleanup_tests tests/cleanup_tests.cpp)
target_include_directories(cleanup_tests PRIVATE include)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_file_io file_io_tests cleanup_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endforeach()
endif()
add_test(NAME checked_file_io COMMAND file_io_tests)
add_test(NAME checked_resource_cleanup COMMAND cleanup_tests)

add_library(saccade_model_manifest src/model_manifest.cpp)
target_include_directories(saccade_model_manifest PUBLIC include)
target_link_libraries(saccade_model_manifest PUBLIC saccade_file_io saccade_sha256)
add_executable(model_manifest tools/model_manifest.cpp)
target_link_libraries(model_manifest PRIVATE saccade_model_manifest)
add_executable(model_manifest_tests tests/model_manifest_tests.cpp)
target_link_libraries(model_manifest_tests PRIVATE saccade_model_manifest)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_model_manifest model_manifest model_manifest_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endforeach()
endif()
add_test(NAME model_manifest_contract COMMAND model_manifest_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model-manifest)
add_test(NAME model_manifest_cli
  COMMAND ${CMAKE_COMMAND}
    -DPROGRAM=$<TARGET_FILE:model_manifest>
    -DFIXTURE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/model-manifest
    -DFIXTURE_DIR=${CMAKE_CURRENT_BINARY_DIR}/manifest-fixtures
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ManifestTest.cmake)

# S01 scalar reference for the m head's saccade::SelectiveScan and Cross-Scan.
# binary64 throughout; contraction is disabled so the reference does not depend
# on the target's FMA availability (docs/SCAN_CONTRACT.md §5).
add_library(saccade_selective_scan src/selective_scan.cpp)
target_include_directories(saccade_selective_scan PUBLIC include)
add_executable(selective_scan_tests tests/selective_scan_tests.cpp)
target_link_libraries(selective_scan_tests PRIVATE saccade_selective_scan saccade_file_io)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_selective_scan selective_scan_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endforeach()
endif()
add_test(NAME selective_scan_reference COMMAND selective_scan_tests
  ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/selective-scan)

# S04 native weight delivery of the m head: a code-free torch.save zip/pickle
# reader, the head parameter schema and the canonical bundle (docs/HEAD_TRT.md §2).
add_library(saccade_head_weights src/torch_checkpoint.cpp src/head_weights.cpp)
target_include_directories(saccade_head_weights PUBLIC include)
target_link_libraries(saccade_head_weights PUBLIC saccade_sha256)
add_executable(head_weights tools/head_weights.cpp)
target_link_libraries(head_weights PRIVATE saccade_head_reference saccade_file_io)
# Binary64 reference of the complete head with its running error bound
# (docs/HEAD_TRT.md §4); no contraction, as for the S01 reference.
add_library(saccade_head_reference src/head_reference.cpp src/head_compare.cpp)
target_link_libraries(saccade_head_reference PUBLIC saccade_head_weights)
find_package(Threads REQUIRED)
target_link_libraries(saccade_head_reference PRIVATE Threads::Threads)
add_executable(head_weights_tests tests/head_weights_tests.cpp)
target_link_libraries(head_weights_tests PRIVATE saccade_head_reference)
add_executable(head_reference_tests tests/head_reference_tests.cpp)
target_link_libraries(head_reference_tests PRIVATE saccade_head_reference saccade_selective_scan saccade_file_io)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_head_weights head_weights saccade_head_reference head_weights_tests head_reference_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endforeach()
endif()
add_test(NAME head_weight_delivery COMMAND head_weights_tests)
add_test(NAME head_reference COMMAND head_reference_tests)
set_tests_properties(head_reference PROPERTIES TIMEOUT 900)
# The real candidate (docs/HEAD_TRT.md §2) is not in the repository: when its
# checkpoint is given, convert it and require the recorded hashes.
set(SACCADE_HEAD_CHECKPOINT "" CACHE FILEPATH "T1 head checkpoint best.ckpt of the S04 candidate (optional)")
# Defaults are the S04 candidate; another candidate (T01, docs/HEAD_TRT.md §10)
# is configured with its own recorded hashes.
set(SACCADE_HEAD_CHECKPOINT_SHA256 "a5a0e7091cd854ad68210ae4bbd43af73c7d3d098ea119f02ea65fab59c0676c"
  CACHE STRING "SHA-256 of SACCADE_HEAD_CHECKPOINT")
set(SACCADE_HEAD_BUNDLE_SHA256 "e1a8660fb9ad510f20ccc3d56a4feac61c6a2e5f771aab531caa741e7677f310"
  CACHE STRING "SHA-256 of the SCWB bundle converted from SACCADE_HEAD_CHECKPOINT")
if(SACCADE_HEAD_CHECKPOINT)
  set(SACCADE_HEAD_TEST_BUNDLE "${CMAKE_CURRENT_BINARY_DIR}/head-weights/t1-head.swb")
  add_test(NAME head_weights_candidate
    COMMAND ${CMAKE_COMMAND} -DPROGRAM=$<TARGET_FILE:head_weights> -DCHECKPOINT=${SACCADE_HEAD_CHECKPOINT}
      -DCHECKPOINT_SHA256=${SACCADE_HEAD_CHECKPOINT_SHA256} -DBUNDLE_SHA256=${SACCADE_HEAD_BUNDLE_SHA256}
      -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/head-weights -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeadWeightsTest.cmake)
  set_tests_properties(head_weights_candidate PROPERTIES FIXTURES_SETUP head_bundle TIMEOUT 300)
  add_test(NAME head_reference_candidate COMMAND head_reference_tests "${SACCADE_HEAD_TEST_BUNDLE}" ${SACCADE_HEAD_BUNDLE_SHA256})
  set_tests_properties(head_reference_candidate PROPERTIES FIXTURES_REQUIRED head_bundle TIMEOUT 900)
endif()

include(cmake/TensorRT.cmake)

# F04 scalar reference and device-independent candidate ABI. The default build
# exercises the observation policy without requiring a CUDA SDK.
add_library(saccade_postprocess src/postprocess.cpp)
target_include_directories(saccade_postprocess PUBLIC include)
add_executable(postprocess_tests tests/postprocess_tests.cpp)
target_link_libraries(postprocess_tests PRIVATE saccade_postprocess)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_postprocess postprocess_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endforeach()
endif()
add_test(NAME postprocess_reference COMMAND postprocess_tests)

option(SACCADE_ENABLE_CUDA "Build the synthetic CUDA protocol executor and GPU tests" OFF)
if(SACCADE_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SACCADE_BUILD_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND git status --porcelain WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE SACCADE_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  if(SACCADE_GIT_STATUS STREQUAL "")
    set(SACCADE_BUILD_DIRTY "false")
  else()
    set(SACCADE_BUILD_DIRTY "true")
  endif()
  add_library(saccade_synthetic_cuda cuda/synthetic_runtime.cu)
  set_target_properties(saccade_synthetic_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(saccade_synthetic_cuda PRIVATE
    SACCADE_BUILD_COMMIT="${SACCADE_BUILD_COMMIT}" SACCADE_BUILD_DIRTY="${SACCADE_BUILD_DIRTY}")
  target_compile_options(saccade_synthetic_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  target_link_libraries(saccade_synthetic_cuda PUBLIC saccade_protocol saccade_synthetic_config CUDA::cudart CUDA::cufft)
  add_executable(cuda_synthetic tools/cuda_synthetic.cpp)
  add_executable(cuda_synthetic_tests tests/cuda_synthetic_tests.cpp)
  target_link_libraries(cuda_synthetic PRIVATE saccade_synthetic_cuda saccade_file_io)
  target_link_libraries(cuda_synthetic_tests PRIVATE saccade_synthetic_cuda saccade_file_io)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cuda_synthetic PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(cuda_synthetic_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  # S02 CUDA FP32 forward of the S01 scan contract (docs/SCAN_CUDA.md §3): no
  # fast math and subnormals kept; the kernels spell out every rounding with
  # __*_rn intrinsics, so contraction cannot change the scan's results.
  set(SACCADE_SCAN_CUDA_FLAGS --ftz=false --prec-div=true --prec-sqrt=true --fmad=true -lineinfo)
  string(JOIN " " SACCADE_SCAN_CUDA_FLAGS_TEXT ${SACCADE_SCAN_CUDA_FLAGS})
  add_library(saccade_selective_scan_cuda cuda/selective_scan.cu)
  set_target_properties(saccade_selective_scan_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(saccade_selective_scan_cuda PRIVATE
    SACCADE_BUILD_COMMIT="${SACCADE_BUILD_COMMIT}" SACCADE_BUILD_DIRTY="${SACCADE_BUILD_DIRTY}"
    SACCADE_SCAN_CUDA_FLAGS="${SACCADE_SCAN_CUDA_FLAGS_TEXT}")
  target_compile_options(saccade_selective_scan_cuda PRIVATE ${SACCADE_SCAN_CUDA_FLAGS})
  target_link_libraries(saccade_selective_scan_cuda PUBLIC saccade_selective_scan CUDA::cudart)
  add_executable(selective_scan_cuda_tests tests/selective_scan_cuda_tests.cpp)
  target_link_libraries(selective_scan_cuda_tests PRIVATE saccade_selective_scan_cuda saccade_file_io)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(selective_scan_cuda_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endif()
  add_test(NAME selective_scan_cuda COMMAND selective_scan_cuda_tests)
  set_tests_properties(selective_scan_cuda PROPERTIES LABELS "gpu" TIMEOUT 300)
  # Kernels are loaded explicitly, so the held-stream test also passes under lazy loading.
  add_test(NAME selective_scan_cuda_lazy_streams COMMAND selective_scan_cuda_tests --group streams)
  set_tests_properties(selective_scan_cuda_lazy_streams PROPERTIES LABELS "gpu" TIMEOUT 120
    ENVIRONMENT CUDA_MODULE_LOADING=LAZY)
  add_test(NAME cuda_synthetic_protocol COMMAND cuda_synthetic_tests ${CMAKE_CURRENT_BINARY_DIR}/gpu-evidence)
  set_tests_properties(cuda_synthetic_protocol PROPERTIES LABELS "gpu" TIMEOUT 180)
  add_test(NAME cuda_synthetic_cli_io
    COMMAND ${CMAKE_COMMAND}
      -DPROGRAM=$<TARGET_FILE:cuda_synthetic>
      -DFIXTURE_DIR=${CMAKE_CURRENT_BINARY_DIR}/synthetic-io-fixtures
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SyntheticIOTest.cmake)
  set_tests_properties(cuda_synthetic_cli_io PROPERTIES LABELS "gpu" TIMEOUT 180)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_executable(cuda_cleanup_tests tests/cuda_cleanup_tests.cpp)
    target_link_libraries(cuda_cleanup_tests PRIVATE saccade_synthetic_cuda)
    target_compile_options(cuda_cleanup_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
    foreach(api cufftDestroy cudaFree cudaFreeHost cudaEventDestroy cudaStreamDestroy)
      target_link_options(cuda_cleanup_tests PRIVATE "-Wl,--wrap=${api}")
    endforeach()
    add_test(NAME cuda_checked_cleanup COMMAND cuda_cleanup_tests)
    set_tests_properties(cuda_checked_cleanup PROPERTIES LABELS "gpu" TIMEOUT 180)
  endif()
endif()

# S03 TensorRT scan plugins, minimal engines and frame harness: they need the
# S02 CUDA library and the pinned native TensorRT SDK (docs/SCAN_TRT.md).
if(SACCADE_ENABLE_CUDA AND SACCADE_ENABLE_TENSORRT)
  include(cmake/ScanTensorRT.cmake)
elseif(SACCADE_ENABLE_TENSORRT)
  message(STATUS "S03 scan TensorRT plugins need SACCADE_ENABLE_CUDA=ON; not built")
endif()

# S04 complete head: native TensorRT network from the weight bundle, the
# frame-session lifecycle tool and the GPU acceptance (docs/HEAD_TRT.md).
if(SACCADE_ENABLE_CUDA AND SACCADE_ENABLE_TENSORRT)
  include(cmake/HeadTensorRT.cmake)
endif()

# T01 backbone/neck and F03 complete model: the backbone ONNX through the pinned
# SDK's ONNX parser, the S04 head on its outputs (docs/BACKBONE_POLICY.md).
option(SACCADE_ENABLE_MODEL_TRT "Build the T01 backbone and F03 model engine tools (needs the SDK's ONNX parser)" OFF)
if(SACCADE_ENABLE_MODEL_TRT)
  if(NOT (SACCADE_ENABLE_CUDA AND SACCADE_ENABLE_TENSORRT))
    message(FATAL_ERROR "SACCADE_ENABLE_MODEL_TRT needs SACCADE_ENABLE_CUDA=ON and SACCADE_ENABLE_TENSORRT=ON")
  endif()
  include(cmake/ModelTensorRT.cmake)
endif()

# Native F01/F02 is independent of TensorRT. The default CPU build stays SDK-free.
option(SACCADE_ENABLE_PREPROCESS "Build hardware nvJPEG and CUDA antialiased resize" OFF)
if(SACCADE_ENABLE_PREPROCESS)
  if(NOT SACCADE_ENABLE_CUDA)
    message(FATAL_ERROR "SACCADE_ENABLE_PREPROCESS needs SACCADE_ENABLE_CUDA=ON")
  endif()
  include(cmake/Preprocess.cmake)
endif()

# F04 has no TensorRT or nvJPEG dependency: it consumes recorded or live raw
# tensors plus the producer's device status on the caller's explicit stream.
option(SACCADE_ENABLE_POSTPROCESS "Build F04 CUDA detection postprocessing" OFF)
if(SACCADE_ENABLE_POSTPROCESS)
  if(NOT SACCADE_ENABLE_CUDA)
    message(FATAL_ERROR "SACCADE_ENABLE_POSTPROCESS needs SACCADE_ENABLE_CUDA=ON")
  endif()
  include(cmake/Postprocess.cmake)
endif()
