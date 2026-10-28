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

add_library(saccade_synthetic_config src/synthetic_config.cpp)
target_include_directories(saccade_synthetic_config PUBLIC include)
add_executable(synthetic_config_tests tests/synthetic_config_tests.cpp)
target_link_libraries(synthetic_config_tests PRIVATE saccade_synthetic_config)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(saccade_synthetic_config PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_compile_options(synthetic_config_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME synthetic_resolved_config COMMAND synthetic_config_tests)

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
  target_link_libraries(cuda_synthetic PRIVATE saccade_synthetic_cuda)
  target_link_libraries(cuda_synthetic_tests PRIVATE saccade_synthetic_cuda)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(cuda_synthetic PRIVATE -Wall -Wextra -Wpedantic -Werror)
    target_compile_options(cuda_synthetic_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME cuda_synthetic_protocol COMMAND cuda_synthetic_tests ${CMAKE_CURRENT_BINARY_DIR}/gpu-evidence)
  set_tests_properties(cuda_synthetic_protocol PROPERTIES LABELS "gpu" TIMEOUT 180)
endif()
