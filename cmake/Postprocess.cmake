add_library(saccade_postprocess_cuda cuda/postprocess.cu)
set_target_properties(saccade_postprocess_cuda PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_include_directories(saccade_postprocess_cuda PUBLIC include)
target_link_libraries(saccade_postprocess_cuda PUBLIC saccade_postprocess CUDA::cudart CUDA::cuda_driver)
target_compile_options(saccade_postprocess_cuda PRIVATE
  --ftz=false --prec-div=true --prec-sqrt=true --fmad=false -lineinfo)
target_compile_definitions(saccade_postprocess_cuda PRIVATE
  SACCADE_BUILD_COMMIT="${SACCADE_BUILD_COMMIT}" SACCADE_BUILD_DIRTY="${SACCADE_BUILD_DIRTY}"
  SACCADE_POSTPROCESS_CUDA_FLAGS="--ftz=false --prec-div=true --prec-sqrt=true --fmad=false -lineinfo")
add_executable(postprocess_cuda_tests tests/postprocess_cuda_tests.cpp)
target_link_libraries(postprocess_cuda_tests PRIVATE
  saccade_postprocess_cuda saccade_file_io saccade_sha256 Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(postprocess_cuda_tests PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
endif()
add_test(NAME postprocess_cuda COMMAND postprocess_cuda_tests)
set_tests_properties(postprocess_cuda PROPERTIES LABELS "gpu;postprocess" TIMEOUT 300)
add_test(NAME postprocess_cuda_lazy_streams COMMAND postprocess_cuda_tests --streams-only)
set_tests_properties(postprocess_cuda_lazy_streams PROPERTIES LABELS "gpu;postprocess" TIMEOUT 90
  ENVIRONMENT CUDA_MODULE_LOADING=LAZY)
# Deliberately faulty positive controls, run only by the evidence script under
# the matching sanitizer in their own processes; never part of normal CTest.
add_executable(postprocess_sanitizer_control tests/postprocess_sanitizer_control.cu)
set_target_properties(postprocess_sanitizer_control PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_link_libraries(postprocess_sanitizer_control PRIVATE CUDA::cudart)
target_compile_options(postprocess_sanitizer_control PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
set(SACCADE_POSTPROCESS_F03_DUMPS "" CACHE PATH "Recorded F03 model-dump directory (optional, hashes are pinned)")
if(SACCADE_POSTPROCESS_F03_DUMPS)
  add_test(NAME postprocess_f03_replay COMMAND postprocess_cuda_tests
    --f03-dumps ${SACCADE_POSTPROCESS_F03_DUMPS})
  set_tests_properties(postprocess_f03_replay PROPERTIES LABELS "gpu;postprocess" TIMEOUT 300)
endif()
