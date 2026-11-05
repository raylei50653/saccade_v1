# M1 offline harness: recorded detections + motion -> CPU tracker -> MOT file.
add_executable(track_offline tools/track_offline.cpp)
target_link_libraries(track_offline PRIVATE saccade_tracking_reference saccade_motion saccade_mot_output)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(track_offline PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
endif()

# CPU research only: optional occlusion costs against fixed detection/motion
# records. Native tracker defaults and CUDA state/config are unchanged.
add_library(saccade_occlusion_experiment src/occlusion_experiment.cpp)
target_link_libraries(saccade_occlusion_experiment PUBLIC saccade_tracking_reference)
add_executable(occlusion_ablation tools/occlusion_ablation.cpp)
target_link_libraries(occlusion_ablation PRIVATE saccade_occlusion_experiment saccade_motion saccade_mot_output saccade_file_io)
add_executable(occlusion_experiment_tests tests/occlusion_experiment_tests.cpp)
target_link_libraries(occlusion_experiment_tests PRIVATE saccade_occlusion_experiment)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_occlusion_experiment occlusion_ablation occlusion_experiment_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
  endforeach()
endif()
add_test(NAME occlusion_experiment COMMAND occlusion_experiment_tests)
add_test(NAME occlusion_ablation_cli
  COMMAND ${CMAKE_COMMAND}
    -DPROGRAM=$<TARGET_FILE:occlusion_ablation>
    -DFIXTURE_DIR=${CMAKE_CURRENT_BINARY_DIR}/occlusion-cli-fixtures
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OcclusionAblationCliTest.cmake)
