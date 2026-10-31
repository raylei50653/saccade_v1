add_library(saccade_model_producer_cuda cuda/model_producer.cu)
set_target_properties(saccade_model_producer_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(saccade_model_producer_cuda PUBLIC saccade_postprocess CUDA::cudart CUDA::cuda_driver)
target_compile_options(saccade_model_producer_cuda PRIVATE --ftz=false --fmad=false -lineinfo)
target_compile_definitions(saccade_model_producer_cuda PRIVATE
  SACCADE_BUILD_COMMIT="${SACCADE_BUILD_COMMIT}" SACCADE_BUILD_DIRTY="${SACCADE_BUILD_DIRTY}")

if(SACCADE_ENABLE_POSTPROCESS)
  add_executable(model_producer_cuda_tests tests/model_producer_cuda_tests.cpp tests/scan_trt_api_trace.cpp)
  target_link_libraries(model_producer_cuda_tests PRIVATE
    saccade_model_producer_cuda saccade_postprocess_cuda CUDA::cupti Threads::Threads)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(model_producer_cuda_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endif()
  add_test(NAME model_producer_cuda COMMAND model_producer_cuda_tests)
  add_test(NAME model_producer_cuda_lazy_streams COMMAND model_producer_cuda_tests --streams-only)
  set_tests_properties(model_producer_cuda model_producer_cuda_lazy_streams PROPERTIES
    LABELS "gpu;model_producer" TIMEOUT 180)
  set_tests_properties(model_producer_cuda_lazy_streams PROPERTIES ENVIRONMENT CUDA_MODULE_LOADING=LAZY)

  if(SACCADE_ENABLE_MODEL_TRT)
    add_executable(model_producer_trt_tests tests/model_producer_trt_tests.cpp)
    target_link_libraries(model_producer_trt_tests PRIVATE
      saccade_model_trt saccade_model_producer_cuda saccade_postprocess_cuda saccade_file_io saccade_sha256)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(model_producer_trt_tests PRIVATE ${SACCADE_SCAN_TRT_WARNINGS} -ffp-contract=off)
    endif()
    set_target_properties(model_producer_trt_tests PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
    target_link_options(model_producer_trt_tests PRIVATE "LINKER:--disable-new-dtags")
    set(SACCADE_MODEL_PRODUCER_ENGINE "" CACHE FILEPATH "Pinned F03 engine for live producer integration (optional)")
    set(SACCADE_MODEL_PRODUCER_INPUTS "" CACHE PATH "Pinned F03 input directory for live producer integration (optional)")
    if(SACCADE_MODEL_PRODUCER_ENGINE AND SACCADE_MODEL_PRODUCER_INPUTS)
      add_test(NAME model_producer_trt COMMAND model_producer_trt_tests
        ${SACCADE_MODEL_PRODUCER_ENGINE} ${SACCADE_MODEL_PRODUCER_INPUTS})
      set_tests_properties(model_producer_trt PROPERTIES LABELS "gpu;model_producer;tensorrt" TIMEOUT 600)
    endif()
  endif()
endif()
