# S04 complete T=1 head on TensorRT (docs/HEAD_TRT.md). Included after
# cmake/ScanTensorRT.cmake, with the S03 plugins and frame harness available.
add_library(saccade_head_trt src/head_trt_network.cpp)
target_link_libraries(saccade_head_trt PUBLIC saccade_scan_trt_frame saccade_head_reference)

add_executable(head_trt tools/head_trt.cpp)
target_link_libraries(head_trt PRIVATE saccade_head_trt saccade_file_io saccade_sha256)

foreach(target saccade_head_trt head_trt)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE ${SACCADE_SCAN_TRT_WARNINGS})
  endif()
endforeach()
set_target_properties(head_trt PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
target_link_options(head_trt PRIVATE "LINKER:--disable-new-dtags")

add_executable(head_trt_tests tests/head_trt_tests.cpp tests/scan_trt_api_trace.cpp)
target_link_libraries(head_trt_tests PRIVATE saccade_head_trt saccade_file_io saccade_sha256 CUDA::cupti)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(head_trt_tests PRIVATE ${SACCADE_SCAN_TRT_WARNINGS} -ffp-contract=off)
endif()
set_target_properties(head_trt_tests PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
target_link_options(head_trt_tests PRIVATE "LINKER:--disable-new-dtags")

# With the candidate checkpoint configured the engine is built from its bundle
# (head_weights_candidate); otherwise from synthetic weights, so the structure
# and failure paths still run (the numeric verdict then concerns no model).
set(SACCADE_HEAD_TRT_WORK "${CMAKE_CURRENT_BINARY_DIR}/head-trt")
set(SACCADE_HEAD_TRT_ARGS -DWORK=${SACCADE_HEAD_TRT_WORK} -DHEAD_TRT=$<TARGET_FILE:head_trt> -DWEIGHTS=$<TARGET_FILE:head_weights>
  -DSUITE=$<TARGET_FILE:head_trt_tests>)
if(SACCADE_HEAD_CHECKPOINT)
  list(APPEND SACCADE_HEAD_TRT_ARGS -DCANDIDATE_BUNDLE=${SACCADE_HEAD_TEST_BUNDLE})
endif()
add_test(NAME head_trt_engine_build COMMAND ${CMAKE_COMMAND} ${SACCADE_HEAD_TRT_ARGS} -DMODE=build
  -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeadTrtTest.cmake)
set_tests_properties(head_trt_engine_build PROPERTIES LABELS "gpu;tensorrt" TIMEOUT 600 FIXTURES_SETUP head_trt_engine)
if(SACCADE_HEAD_CHECKPOINT)
  set_tests_properties(head_trt_engine_build PROPERTIES FIXTURES_REQUIRED head_bundle)
endif()
add_test(NAME head_trt_engine_run COMMAND ${CMAKE_COMMAND} ${SACCADE_HEAD_TRT_ARGS} -DMODE=run -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeadTrtTest.cmake)
add_test(NAME head_trt_engine_unregistered COMMAND ${CMAKE_COMMAND} ${SACCADE_HEAD_TRT_ARGS} -DMODE=unregistered
  -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeadTrtTest.cmake)
add_test(NAME head_trt COMMAND ${CMAKE_COMMAND} ${SACCADE_HEAD_TRT_ARGS} -DMODE=suite
  "-DARGS=${SACCADE_HEAD_TRT_WORK}/suite-report.tsv" -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeadTrtTest.cmake)
add_test(NAME head_trt_api_trace COMMAND ${CMAKE_COMMAND} ${SACCADE_HEAD_TRT_ARGS} -DMODE=suite "-DARGS=--group api_trace"
  -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HeadTrtTest.cmake)
set_tests_properties(head_trt_engine_run head_trt_engine_unregistered head_trt head_trt_api_trace PROPERTIES
  LABELS "gpu;tensorrt" TIMEOUT 900 FIXTURES_REQUIRED head_trt_engine)
# A fresh process loads every kernel during initialization, not on the frame path.
set_tests_properties(head_trt_engine_run PROPERTIES ENVIRONMENT CUDA_MODULE_LOADING=LAZY)
