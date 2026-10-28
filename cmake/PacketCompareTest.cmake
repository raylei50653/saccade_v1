execute_process(COMMAND "${PROTOCOL_TESTS}" --fixtures "${FIXTURE_DIR}"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Could not generate packet fixtures: ${output}${error}")
endif()

function(check_compare right expected)
  execute_process(COMMAND "${PACKET_COMPARE}" "${FIXTURE_DIR}/same.bin" "${FIXTURE_DIR}/${right}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT result EQUAL expected)
    message(FATAL_ERROR "packet_compare ${right}: expected ${expected}, got ${result}: ${output}${error}")
  endif()
endfunction()

check_compare(same.bin 0)
check_compare(different-unused.bin 1)
check_compare(different-reserved.bin 1)
check_compare(truncated.bin 2)
check_compare(empty.bin 2)
check_compare(missing.bin 2)
