# CUDA CLI I/O regressions. PROGRAM and FIXTURE_DIR are supplied by CTest.
if(NOT DEFINED PROGRAM OR NOT EXISTS "${PROGRAM}")
  message(FATAL_ERROR "PROGRAM must name the built cuda_synthetic executable")
endif()
if(NOT DEFINED FIXTURE_DIR OR FIXTURE_DIR STREQUAL "")
  message(FATAL_ERROR "FIXTURE_DIR is required")
endif()

# Every invocation owns a fresh child directory. Keep prior fixtures/evidence;
# never remove a caller-supplied directory or follow its output symlinks.
string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef run_id)
set(run_dir "${FIXTURE_DIR}/io-${run_id}")
file(MAKE_DIRECTORY "${run_dir}" "${run_dir}/directory-config")
file(WRITE "${run_dir}/empty.cfg" "")
file(WRITE "${run_dir}/one-frame.cfg" "frames=1\ndelay_cycles=0\n")

function(expect_io_failure name config output_dir expected_error)
  execute_process(COMMAND "${PROGRAM}" "${config}" "${output_dir}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 45)
  file(WRITE "${run_dir}/${name}.stdout.log" "${output}")
  file(WRITE "${run_dir}/${name}.stderr.log" "${error}")
  file(WRITE "${run_dir}/${name}.exit" "${result}\n")
  # An arbitrary nonzero result could be a GPU/launch failure and is not proof
  # of the intended I/O rejection. Check exact exit status and error context.
  if(NOT "${result}" STREQUAL "2")
    message(FATAL_ERROR "${name}: expected exit 2, got ${result}: ${output}${error}")
  endif()
  string(FIND "${error}" "cuda_synthetic: ${expected_error}" error_offset)
  if(error_offset EQUAL -1)
    message(FATAL_ERROR "${name}: missing expected I/O diagnostic '${expected_error}': ${output}${error}")
  endif()
  message(STATUS "${name}: exit 2 with expected I/O diagnostic")
endfunction()

expect_io_failure(directory_config "${run_dir}/directory-config" "${run_dir}/directory-output"
  "cannot read input: ${run_dir}/directory-config")
expect_io_failure(missing_config "${run_dir}/missing.cfg" "${run_dir}/missing-output"
  "cannot open input: ${run_dir}/missing.cfg")
if(EXISTS "${run_dir}/directory-output" OR EXISTS "${run_dir}/missing-output")
  message(FATAL_ERROR "invalid config input created an output directory before rejection")
endif()

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux" OR NOT EXISTS "/dev/full")
  message(STATUS "SKIP /dev/full CLI write cases: Linux /dev/full is unavailable")
  return()
endif()

function(full_output directory filename)
  file(MAKE_DIRECTORY "${directory}")
  file(CREATE_LINK "/dev/full" "${directory}/${filename}" SYMBOLIC RESULT link_result)
  if(NOT "${link_result}" STREQUAL "0")
    message(FATAL_ERROR "cannot create /dev/full fixture symlink: ${link_result}")
  endif()
endfunction()

# Empty regular input is valid and resolves typed defaults. A failure writing
# the first config artifact happens before a CUDA session can be started.
set(resolved_dir "${run_dir}/resolved-full")
full_output("${resolved_dir}" "resolved.cfg")
expect_io_failure(resolved_write_full "${run_dir}/empty.cfg" "${resolved_dir}"
  "cannot write or close output: ${resolved_dir}/resolved.cfg")
if(EXISTS "${resolved_dir}/packets.bin")
  message(FATAL_ERROR "resolved.cfg failure unexpectedly produced GPU packet output")
endif()

# One successful GPU frame precedes metadata writing. Small identity text may
# stay buffered until close, so this specifically checks close error handling.
set(identity_dir "${run_dir}/identity-full")
full_output("${identity_dir}" "identity.txt")
expect_io_failure(identity_write_full "${run_dir}/one-frame.cfg" "${identity_dir}"
  "cannot write or close output: ${identity_dir}/identity.txt")
if(NOT EXISTS "${identity_dir}/packets.bin")
  message(FATAL_ERROR "metadata failure fixture did not preserve raw GPU packet output")
endif()
file(SIZE "${identity_dir}/packets.bin" packet_size)
if(NOT packet_size EQUAL 12368)
  message(FATAL_ERROR "metadata failure fixture expected one raw GPU packet, got ${packet_size} bytes")
endif()

# A nonempty raw binary packet stream must propagate /dev/full's write error;
# later artifacts must not make the CLI report success or mask the error.
set(packets_dir "${run_dir}/packets-full")
full_output("${packets_dir}" "packets.bin")
expect_io_failure(packet_write_full "${run_dir}/one-frame.cfg" "${packets_dir}"
  "cannot write or close output: ${packets_dir}/packets.bin")
if(EXISTS "${packets_dir}/identity.txt")
  message(FATAL_ERROR "packet write failure unexpectedly continued to identity output")
endif()
message(STATUS "CUDA CLI I/O regressions passed; fixtures: ${run_dir}")
