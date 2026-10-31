# CPU-only finite model; does not claim GPU scheduling coverage.
add_executable(detection_session_model tools/detection_session_model.cpp)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(detection_session_model PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
add_test(NAME detection_session_happens_before COMMAND detection_session_model)
