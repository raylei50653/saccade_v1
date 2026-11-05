# Reject output aliases before input records are opened; no dataset required.
if(NOT DEFINED PROGRAM OR NOT EXISTS "${PROGRAM}")
  message(FATAL_ERROR "PROGRAM must name the built occlusion_ablation executable")
endif()
if(NOT DEFINED FIXTURE_DIR OR FIXTURE_DIR STREQUAL "")
  message(FATAL_ERROR "FIXTURE_DIR is required")
endif()

string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef run_id)
set(run_dir "${FIXTURE_DIR}/${run_id}")
file(MAKE_DIRECTORY "${run_dir}/child")

function(expect_rejection name mot diag arm expected_error)
  execute_process(COMMAND "${PROGRAM}"
    "${run_dir}/missing.scdr" "${run_dir}/missing.scmo" "${mot}" "${arm}" "${diag}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 10)
  if(NOT "${result}" STREQUAL "1")
    message(FATAL_ERROR "${name}: expected exit 1, got ${result}: ${output}${error}")
  endif()
  string(FIND "${error}" "occlusion_ablation: ${expected_error}" error_offset)
  if(error_offset EQUAL -1)
    message(FATAL_ERROR "${name}: missing expected diagnostic '${expected_error}': ${output}${error}")
  endif()
  if(EXISTS "${mot}" OR EXISTS "${diag}")
    message(FATAL_ERROR "${name}: rejection created an output file")
  endif()
  message(STATUS "${name}: exit 1 with expected diagnostic and no output files")
endfunction()

set(same_path_error "experiment outputs must name distinct files")
expect_rejection(equal_paths "${run_dir}/equal.txt" "${run_dir}/equal.txt"
  baseline "${same_path_error}")
expect_rejection(normalized_alias "${run_dir}/normalized.txt" "${run_dir}/child/../normalized.txt"
  baseline "${same_path_error}")

if(CMAKE_HOST_UNIX)
  file(CREATE_LINK "${run_dir}/child" "${run_dir}/linked-child" SYMBOLIC RESULT link_result)
  if(NOT "${link_result}" STREQUAL "0")
    message(FATAL_ERROR "cannot create parent-symlink fixture: ${link_result}")
  endif()
  expect_rejection(parent_symlink_alias "${run_dir}/child/symlink.txt" "${run_dir}/linked-child/symlink.txt"
    baseline "${same_path_error}")
endif()

expect_rejection(invalid_arm "${run_dir}/invalid.txt" "${run_dir}/invalid.json"
  invalid "unknown experiment arm")
