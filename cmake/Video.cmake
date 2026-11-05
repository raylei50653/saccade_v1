option(SACCADE_ENABLE_VIDEO "Build native FFmpeg file/RTSP/camera input (CPU decode)" OFF)
option(SACCADE_ENABLE_NVDEC "Build strict native NVDEC device-surface video decode" OFF)
if(SACCADE_ENABLE_NVDEC AND NOT (SACCADE_ENABLE_VIDEO AND SACCADE_ENABLE_CUDA))
  message(FATAL_ERROR "SACCADE_ENABLE_NVDEC requires SACCADE_ENABLE_VIDEO and SACCADE_ENABLE_CUDA")
endif()
if(NOT SACCADE_ENABLE_VIDEO)
  return()
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(SACCADE_FFMPEG REQUIRED IMPORTED_TARGET
  libavformat libavcodec libavutil libswscale libavdevice)
add_library(saccade_video_source src/video_source.cpp)
target_include_directories(saccade_video_source PUBLIC include)
target_link_libraries(saccade_video_source PUBLIC PkgConfig::SACCADE_FFMPEG Threads::Threads)
if(SACCADE_ENABLE_NVDEC)
  target_compile_definitions(saccade_video_source PRIVATE SACCADE_VIDEO_NVDEC=1)
  target_link_libraries(saccade_video_source PUBLIC CUDA::cuda_driver CUDA::cudart)
endif()
add_executable(video_probe tools/video_probe.cpp)
target_link_libraries(video_probe PRIVATE saccade_video_source)
add_executable(video_source_tests tests/video_source_tests.cpp)
target_link_libraries(video_source_tests PRIVATE saccade_video_source Threads::Threads)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_video_source video_probe video_source_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endforeach()
endif()
find_program(SACCADE_FFMPEG_EXECUTABLE ffmpeg REQUIRED)
set(SACCADE_VIDEO_TEST_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/video-source-bframes.avi")
add_custom_command(OUTPUT "${SACCADE_VIDEO_TEST_FIXTURE}"
  COMMAND "${SACCADE_FFMPEG_EXECUTABLE}" -hide_banner -loglevel error -y
    -f lavfi -i testsrc=size=32x24:rate=10:duration=0.9
    -an -c:v mpeg4 -bf 2 -qscale:v 2 -pix_fmt yuv420p "${SACCADE_VIDEO_TEST_FIXTURE}"
  VERBATIM COMMENT "Generate nine-frame FFmpeg decoder-drain fixture")
add_custom_target(video_source_fixture DEPENDS "${SACCADE_VIDEO_TEST_FIXTURE}")
add_dependencies(video_source_tests video_source_fixture)
add_test(NAME video_source COMMAND video_source_tests "${SACCADE_VIDEO_TEST_FIXTURE}")
set_tests_properties(video_source PROPERTIES TIMEOUT 30)

if(SACCADE_ENABLE_NVDEC)
  set(SACCADE_NVDEC_H264_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/video-nvdec-h264.mp4")
  set(SACCADE_NVDEC_HEVC8_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/video-nvdec-hevc8.mp4")
  set(SACCADE_NVDEC_HEVC10_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/video-nvdec-hevc10.mp4")
  set(SACCADE_NVDEC_UNSUPPORTED_FIXTURE "${CMAKE_CURRENT_BINARY_DIR}/video-nvdec-ffv1.mkv")
  add_custom_command(OUTPUT "${SACCADE_NVDEC_H264_FIXTURE}"
    COMMAND "${SACCADE_FFMPEG_EXECUTABLE}" -hide_banner -loglevel error -y
      -f lavfi -i testsrc2=size=320x192:rate=10:duration=0.9
      -an -c:v libx264 -preset ultrafast -bf 2 -crf 18 -pix_fmt yuv420p
      -color_range tv -colorspace bt709 -color_trc bt709 -color_primaries bt709 "${SACCADE_NVDEC_H264_FIXTURE}"
    VERBATIM COMMENT "Generate H.264 NVDEC B-frame fixture")
  foreach(depth 8 10)
    if(depth EQUAL 8)
      set(nvdec_fixture "${SACCADE_NVDEC_HEVC8_FIXTURE}")
      set(nvdec_pix_fmt yuv420p)
    else()
      set(nvdec_fixture "${SACCADE_NVDEC_HEVC10_FIXTURE}")
      set(nvdec_pix_fmt yuv420p10le)
    endif()
    add_custom_command(OUTPUT "${nvdec_fixture}"
      COMMAND "${SACCADE_FFMPEG_EXECUTABLE}" -hide_banner -loglevel error -y
        -f lavfi -i testsrc2=size=320x192:rate=10:duration=0.9
        -an -c:v libx265 -preset ultrafast -x265-params log-level=error:pools=1:frame-threads=1:bframes=2
        -crf 18 -pix_fmt "${nvdec_pix_fmt}" -color_range tv -colorspace bt709 -color_trc bt709 -color_primaries bt709 "${nvdec_fixture}"
      VERBATIM COMMENT "Generate HEVC ${depth}-bit NVDEC B-frame fixture")
  endforeach()
  add_custom_command(OUTPUT "${SACCADE_NVDEC_UNSUPPORTED_FIXTURE}"
    COMMAND "${SACCADE_FFMPEG_EXECUTABLE}" -hide_banner -loglevel error -y
      -f lavfi -i testsrc2=size=320x192:rate=10:duration=0.9 -an -c:v ffv1 "${SACCADE_NVDEC_UNSUPPORTED_FIXTURE}"
    VERBATIM COMMENT "Generate unsupported-codec no-fallback fixture")
  add_custom_target(video_nvdec_fixtures DEPENDS "${SACCADE_NVDEC_H264_FIXTURE}" "${SACCADE_NVDEC_HEVC8_FIXTURE}"
    "${SACCADE_NVDEC_HEVC10_FIXTURE}" "${SACCADE_NVDEC_UNSUPPORTED_FIXTURE}")
  add_executable(video_source_cuda_tests tests/video_source_cuda_tests.cpp)
  target_link_libraries(video_source_cuda_tests PRIVATE saccade_video_source)
  add_dependencies(video_source_cuda_tests video_nvdec_fixtures)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(video_source_cuda_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  add_test(NAME video_source_cuda COMMAND video_source_cuda_tests "${SACCADE_NVDEC_H264_FIXTURE}"
    "${SACCADE_NVDEC_HEVC8_FIXTURE}" "${SACCADE_NVDEC_HEVC10_FIXTURE}" "${SACCADE_NVDEC_UNSUPPORTED_FIXTURE}")
  set_tests_properties(video_source_cuda PROPERTIES LABELS "gpu;nvdec;streaming" TIMEOUT 120)
endif()
