option(SACCADE_ENABLE_STREAMING "Build live video to native tracking integration" OFF)
if(NOT SACCADE_ENABLE_STREAMING)
  return()
endif()
if(NOT (TARGET saccade_video_source AND TARGET saccade_tracking_session))
  message(FATAL_ERROR "SACCADE_ENABLE_STREAMING needs VIDEO and DETECTION_SESSION (with its CUDA/TensorRT dependencies)")
endif()
if(NOT SACCADE_ENABLE_NVDEC)
  message(FATAL_ERROR "SACCADE_ENABLE_STREAMING requires SACCADE_ENABLE_NVDEC=ON for GPU decoding")
endif()

add_library(saccade_video_view src/video_view.cpp cuda/video_overlay.cu)
target_include_directories(saccade_video_view PUBLIC include)
target_link_libraries(saccade_video_view PUBLIC PkgConfig::SACCADE_FFMPEG CUDA::cuda_driver CUDA::cudart)
set_target_properties(saccade_video_view PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
target_compile_options(saccade_video_view PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
add_executable(video_overlay_tests tests/video_overlay_tests.cpp)
target_link_libraries(video_overlay_tests PRIVATE saccade_video_view)
add_test(NAME video_overlay COMMAND video_overlay_tests)
set_tests_properties(video_overlay PROPERTIES LABELS "gpu;streaming" TIMEOUT 300)

add_executable(track_stream tools/track_stream.cpp)
target_link_libraries(track_stream PRIVATE saccade_video_source saccade_video_view saccade_tracking_session saccade_mot_output
  saccade_resolved_config Threads::Threads)
target_compile_definitions(track_stream PRIVATE SACCADE_BUILD_COMMIT="${SACCADE_P08_COMMIT}"
  SACCADE_BUILD_DIRTY="${SACCADE_P08_DIRTY}")
set_target_properties(track_stream PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
target_link_options(track_stream PRIVATE "LINKER:--disable-new-dtags")
add_executable(tracking_rgb_session_tests tests/tracking_rgb_session_tests.cpp)
target_link_libraries(tracking_rgb_session_tests PRIVATE saccade_tracking_session saccade_file_io)
set_target_properties(tracking_rgb_session_tests PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
target_link_options(tracking_rgb_session_tests PRIVATE "LINKER:--disable-new-dtags")
add_executable(video_device_session_tests tests/video_device_session_tests.cpp)
target_link_libraries(video_device_session_tests PRIVATE saccade_video_source saccade_tracking_session)
set_target_properties(video_device_session_tests PROPERTIES BUILD_RPATH "${SACCADE_TENSORRT_LIBDIR}")
target_link_options(video_device_session_tests PRIVATE "LINKER:--disable-new-dtags")
if(TARGET video_nvdec_fixtures)
  add_dependencies(video_device_session_tests video_nvdec_fixtures)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(saccade_video_view PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror;-ffp-contract=off>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror=all-warnings;-Xcompiler=-Wall,-Wextra,-Werror>)
  foreach(target track_stream tracking_rgb_session_tests video_device_session_tests video_overlay_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endforeach()
endif()
set(SACCADE_STREAMING_TEST_ENGINE "" CACHE FILEPATH "HR FP16 engine for RGB streaming tests")
set(SACCADE_STREAMING_TEST_SEQUENCE "" CACHE PATH "MOT17-10 sequence used by RGB streaming tests")
if(SACCADE_STREAMING_TEST_ENGINE AND SACCADE_STREAMING_TEST_SEQUENCE)
  add_test(NAME video_device_session COMMAND video_device_session_tests
    "${SACCADE_STREAMING_TEST_ENGINE}" "${SACCADE_NVDEC_H264_FIXTURE}")
  set_tests_properties(video_device_session PROPERTIES LABELS "gpu;streaming;session" TIMEOUT 300)
  add_test(NAME tracking_rgb_session COMMAND tracking_rgb_session_tests
    "${SACCADE_STREAMING_TEST_ENGINE}" "${SACCADE_STREAMING_TEST_SEQUENCE}")
  set_tests_properties(tracking_rgb_session PROPERTIES LABELS "gpu;streaming;session" TIMEOUT 300)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    find_program(SACCADE_FFPROBE_EXECUTABLE ffprobe REQUIRED)
    add_test(NAME track_stream_cli COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/track_stream_cli_tests.py"
      "$<TARGET_FILE:track_stream>" "${SACCADE_STREAMING_TEST_ENGINE}" "${SACCADE_STREAMING_TEST_SEQUENCE}"
      --output-root "${CMAKE_CURRENT_BINARY_DIR}/cli-tests"
      --nvdec-fixture "${SACCADE_NVDEC_H264_FIXTURE}"
      --ffmpeg "${SACCADE_FFMPEG_EXECUTABLE}" --ffprobe "${SACCADE_FFPROBE_EXECUTABLE}")
    set_tests_properties(track_stream_cli PROPERTIES LABELS "gpu;streaming;cli" TIMEOUT 300)
    add_test(NAME tracking_p08_native_cli COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/p08_native_cli_tests.py"
      "$<TARGET_FILE:track_stream>" "$<TARGET_FILE:track_sequence>" "$<TARGET_FILE:tracking_config>"
      "${SACCADE_STREAMING_TEST_ENGINE}" "${SACCADE_STREAMING_TEST_SEQUENCE}" "${SACCADE_NVDEC_H264_FIXTURE}"
      --output-root "${CMAKE_CURRENT_BINARY_DIR}/p08-cli-tests")
    set_tests_properties(tracking_p08_native_cli PROPERTIES LABELS "gpu;streaming;cli;p08" TIMEOUT 300)
  endif()
endif()
