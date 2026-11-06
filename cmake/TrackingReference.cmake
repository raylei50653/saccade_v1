# Host numerical policy also applies to consumers that compile replay code.
# CUDA targets retain their explicit device and host compiler flags.
add_library(saccade_tracking_numeric_policy INTERFACE)
target_compile_options(saccade_tracking_numeric_policy INTERFACE
  $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-ffp-contract=off>)

# Configuration and logical types needed by both CPU and native CUDA tracking.
add_library(saccade_tracking_core src/tracking_config.cpp)
target_link_libraries(saccade_tracking_core PUBLIC
  saccade_protocol saccade_postprocess saccade_tracking_numeric_policy)
target_include_directories(saccade_tracking_core PUBLIC include)

# F08a/F04b/F07/F08b serial CPU reference (docs/TRACKING.md).
add_library(saccade_tracking_reference src/tracking_reference.cpp)
target_link_libraries(saccade_tracking_reference PUBLIC saccade_tracking_core)
add_executable(tracking_reference_tests tests/tracking_reference_tests.cpp)
target_link_libraries(tracking_reference_tests PRIVATE saccade_tracking_reference saccade_sha256)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_tracking_core saccade_tracking_reference tracking_reference_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endforeach()
endif()
add_test(NAME tracking_reference COMMAND tracking_reference_tests)
