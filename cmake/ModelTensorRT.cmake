# T01 backbone/neck and F03 complete model on TensorRT (docs/BACKBONE_POLICY.md).
# Included after cmake/HeadTensorRT.cmake with SACCADE_ENABLE_MODEL_TRT: the
# same pinned SDK root must also hold its ONNX parser (docs/NATIVE_TENSORRT.md
# "ONNX parser"); nothing is searched elsewhere.
unset(SACCADE_TENSORRT_ONNX_PARSER_INCLUDE CACHE)
unset(SACCADE_TENSORRT_ONNX_PARSER_LIBRARY CACHE)
find_path(SACCADE_TENSORRT_ONNX_PARSER_INCLUDE NvOnnxParser.h
  PATHS "${SACCADE_TENSORRT_ROOT}/include" "${SACCADE_TENSORRT_ROOT}/usr/include/x86_64-linux-gnu"
  NO_DEFAULT_PATH REQUIRED)
find_file(SACCADE_TENSORRT_ONNX_PARSER_LIBRARY NAMES libnvonnxparser.so.10.16.1
  PATHS "${SACCADE_TENSORRT_ROOT}/lib" "${SACCADE_TENSORRT_ROOT}/lib64"
        "${SACCADE_TENSORRT_ROOT}/usr/lib/x86_64-linux-gnu" NO_DEFAULT_PATH REQUIRED)
get_filename_component(SACCADE_TENSORRT_ONNX_PARSER_LIBRARY "${SACCADE_TENSORRT_ONNX_PARSER_LIBRARY}" REALPATH)
if(NOT SACCADE_TENSORRT_ONNX_PARSER_INCLUDE STREQUAL SACCADE_TENSORRT_INCLUDE)
  message(FATAL_ERROR "ONNX parser header outside the pinned SDK's include directory: ${SACCADE_TENSORRT_ONNX_PARSER_INCLUDE}")
endif()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/onnx-parser-probe.cpp" [=[
#include <NvOnnxParser.h>
#include <iostream>
int main() {
  if (getNvOnnxParserVersion() != NV_ONNX_PARSER_VERSION) {
    std::cerr << "ONNX parser header/library API mismatch: " << getNvOnnxParserVersion() << " vs " << NV_ONNX_PARSER_VERSION << '\n'; return 1;
  }
  std::cout << "nvonnxparser=" << getNvOnnxParserVersion() << '\n';
}
]=])
try_run(SACCADE_ONNX_PARSER_PROBE_EXIT SACCADE_ONNX_PARSER_PROBE_COMPILED
  "${CMAKE_CURRENT_BINARY_DIR}/onnx-parser-probe" "${CMAKE_CURRENT_BINARY_DIR}/onnx-parser-probe.cpp"
  CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${SACCADE_TENSORRT_INCLUDE};${CUDAToolkit_INCLUDE_DIRS}"
  LINK_LIBRARIES "${SACCADE_TENSORRT_ONNX_PARSER_LIBRARY}" "${SACCADE_TENSORRT_LIBRARY}" CUDA::cudart
  COMPILE_OUTPUT_VARIABLE SACCADE_ONNX_PARSER_PROBE_BUILD RUN_OUTPUT_VARIABLE SACCADE_ONNX_PARSER_PROBE_OUTPUT)
if(NOT SACCADE_ONNX_PARSER_PROBE_COMPILED OR NOT SACCADE_ONNX_PARSER_PROBE_EXIT EQUAL 0)
  message(FATAL_ERROR "ONNX parser probe failed; no fallback.\n${SACCADE_ONNX_PARSER_PROBE_BUILD}\n${SACCADE_ONNX_PARSER_PROBE_OUTPUT}")
endif()
message(STATUS "ONNX parser probe: ${SACCADE_ONNX_PARSER_PROBE_OUTPUT}")

add_library(saccade_model_trt src/model_trt_network.cpp)
target_link_libraries(saccade_model_trt PUBLIC saccade_head_trt "${SACCADE_TENSORRT_ONNX_PARSER_LIBRARY}")

add_executable(backbone_trt tools/backbone_trt.cpp)
target_link_libraries(backbone_trt PRIVATE saccade_model_trt saccade_file_io saccade_sha256)
add_executable(model_trt tools/model_trt.cpp)
target_link_libraries(model_trt PRIVATE saccade_model_trt saccade_file_io saccade_sha256)

foreach(target saccade_model_trt backbone_trt model_trt)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE ${SACCADE_SCAN_TRT_WARNINGS})
  endif()
endforeach()
foreach(target backbone_trt model_trt)
  # TensorRT dlopens its per-SM builder resource by basename (see TensorRT.cmake).
  set_target_properties(${target} PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
  target_link_options(${target} PRIVATE "LINKER:--disable-new-dtags")
endforeach()
