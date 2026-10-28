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

include(cmake/TensorRT.cmake)

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
