# M1 offline harness: recorded detections + motion -> CPU tracker -> MOT file.
add_executable(track_offline tools/track_offline.cpp)
target_link_libraries(track_offline PRIVATE saccade_tracking_reference saccade_motion saccade_mot_output
  saccade_resolved_config saccade_file_io)
target_compile_definitions(track_offline PRIVATE SACCADE_BUILD_COMMIT="${SACCADE_P08_COMMIT}"
  SACCADE_BUILD_DIRTY="${SACCADE_P08_DIRTY}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(track_offline PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
endif()
