cmake_minimum_required(VERSION 3.20)
project(saccade_v1_protocol LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(saccade_protocol src/protocol.cpp)
target_include_directories(saccade_protocol PUBLIC include)

add_executable(protocol_model tools/protocol_model.cpp)
add_executable(packet_compare tools/packet_compare.cpp)
target_link_libraries(packet_compare PRIVATE saccade_protocol)
add_executable(protocol_tests tests/protocol_tests.cpp)
target_link_libraries(protocol_tests PRIVATE saccade_protocol)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  foreach(target saccade_protocol protocol_model packet_compare protocol_tests)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endforeach()
endif()

enable_testing()
add_test(NAME protocol_happens_before COMMAND protocol_model --self-test)
add_test(NAME protocol_state_and_packet COMMAND protocol_tests)
add_test(NAME packet_compare_cli
  COMMAND ${CMAKE_COMMAND}
    -DPROTOCOL_TESTS=$<TARGET_FILE:protocol_tests>
    -DPACKET_COMPARE=$<TARGET_FILE:packet_compare>
    -DFIXTURE_DIR=${CMAKE_CURRENT_BINARY_DIR}/packet-fixtures
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/PacketCompareTest.cmake)
