# Pin the host compiler to the self-built GCC 16.2.0 (scripts/build_gcc.sh).
#
# CMakeLists.txt includes this file before project(): unless the build opts out,
# it selects cmake/toolchain-gcc-16.2.0.cmake, which sets both the C++ compiler
# and nvcc's host compiler. After project() and after enable_language(CUDA),
# saccade_check_pinned_compiler() checks what CMake actually picked.
#
# Opt out (other compilers, CI without the pinned GCC):
#   -DSACCADE_ALLOW_UNPINNED_COMPILER=ON
set(SACCADE_PINNED_GCC_VERSION 16.2.0)
if(DEFINED ENV{SACCADE_GCC_ROOT})
  set(SACCADE_PINNED_GCC_ROOT "$ENV{SACCADE_GCC_ROOT}")
else()
  set(SACCADE_PINNED_GCC_ROOT "$ENV{HOME}/.local/opt/gcc-${SACCADE_PINNED_GCC_VERSION}")
endif()
option(SACCADE_ALLOW_UNPINNED_COMPILER
  "Allow a compiler other than the pinned GCC ${SACCADE_PINNED_GCC_VERSION}" OFF)

function(_saccade_real_program name out)
  if(IS_ABSOLUTE "${name}")
    set(path "${name}")
  else()
    find_program(_saccade_program NAMES "${name}")
    set(path "${_saccade_program}")
    unset(_saccade_program CACHE)
  endif()
  if(path AND EXISTS "${path}")
    file(REAL_PATH "${path}" path)
  endif()
  set(${out} "${path}" PARENT_SCOPE)
endfunction()

if(NOT SACCADE_ALLOW_UNPINNED_COMPILER AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  # On a new build tree the toolchain file overrides any compiler named on the
  # command line or in the environment; say so instead of dropping it silently.
  # An existing tree keeps its compiler, and saccade_check_pinned_compiler()
  # reports that instead.
  _saccade_real_program("${SACCADE_PINNED_GCC_ROOT}/bin/g++" _saccade_pinned)
  set(_saccade_requests "")
  if(NOT EXISTS "${CMAKE_BINARY_DIR}/CMakeCache.txt")
    foreach(_saccade_name IN ITEMS CMAKE_CXX_COMPILER CMAKE_CUDA_HOST_COMPILER)
      if(NOT "${${_saccade_name}}" STREQUAL "")
        list(APPEND _saccade_requests "${_saccade_name}=${${_saccade_name}}")
      endif()
    endforeach()
    foreach(_saccade_name IN ITEMS CXX CUDAHOSTCXX)
      if(NOT "$ENV{${_saccade_name}}" STREQUAL "")
        list(APPEND _saccade_requests "${_saccade_name}=$ENV{${_saccade_name}}")
      endif()
    endforeach()
  endif()
  foreach(_saccade_request IN LISTS _saccade_requests)
    string(REGEX REPLACE "^[^=]*=" "" _saccade_value "${_saccade_request}")
    _saccade_real_program("${_saccade_value}" _saccade_requested)
    if(NOT _saccade_requested STREQUAL _saccade_pinned)
      message(WARNING "${_saccade_request} is ignored: this build uses the pinned "
        "GCC ${SACCADE_PINNED_GCC_VERSION}. Pass -DSACCADE_ALLOW_UNPINNED_COMPILER=ON to use it.")
    endif()
  endforeach()
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_LIST_DIR}/toolchain-gcc-${SACCADE_PINNED_GCC_VERSION}.cmake"
    CACHE FILEPATH "Pinned host compiler toolchain")
endif()

function(saccade_check_pinned_compiler)
  set(problems "")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
     OR NOT CMAKE_CXX_COMPILER_VERSION VERSION_EQUAL SACCADE_PINNED_GCC_VERSION)
    list(APPEND problems
      "the C++ compiler is ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (${CMAKE_CXX_COMPILER})")
  endif()
  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if("CUDA" IN_LIST languages)
    if(NOT CMAKE_CUDA_HOST_COMPILER)
      list(APPEND problems "nvcc has no host compiler set and falls back to NVCC_CCBIN or g++ on PATH")
    else()
      _saccade_real_program("${CMAKE_CUDA_HOST_COMPILER}" host)
      _saccade_real_program("${CMAKE_CXX_COMPILER}" cxx)
      if(NOT host STREQUAL cxx)
        list(APPEND problems "nvcc's host compiler ${CMAKE_CUDA_HOST_COMPILER} differs from ${CMAKE_CXX_COMPILER}")
      endif()
    endif()
  endif()
  if(problems)
    list(JOIN problems "; " problems)
    if(SACCADE_ALLOW_UNPINNED_COMPILER)
      message(WARNING "Unpinned compiler (SACCADE_ALLOW_UNPINNED_COMPILER=ON): ${problems}")
    else()
      message(FATAL_ERROR "This build must use the pinned GCC ${SACCADE_PINNED_GCC_VERSION}, but ${problems}. "
        "A build directory configured before the pin keeps its old compiler: configure a new build/<name>. "
        "Install the compiler with scripts/build_gcc.sh, or pass -DSACCADE_ALLOW_UNPINNED_COMPILER=ON "
        "to build with another compiler on purpose.")
    endif()
  endif()
endfunction()
