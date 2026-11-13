# Pinned host compiler: GCC 16.2.0 built by scripts/build_gcc.sh.
#   cmake -S . -B build/<name> -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-gcc-16.2.0.cmake ...
# Set SACCADE_GCC_ROOT to use a different install location.
#
# Only the compiler is pinned. Programs still load the system libstdc++ and
# libgcc_s at run time; those stay backward compatible when the system GCC
# moves to a newer release.
if(DEFINED ENV{SACCADE_GCC_ROOT})
  set(_saccade_gcc_root "$ENV{SACCADE_GCC_ROOT}")
else()
  set(_saccade_gcc_root "$ENV{HOME}/.local/opt/gcc-16.2.0")
endif()

if(NOT EXISTS "${_saccade_gcc_root}/bin/g++")
  message(FATAL_ERROR "GCC 16.2.0 not found at ${_saccade_gcc_root}; run scripts/build_gcc.sh")
endif()

set(CMAKE_C_COMPILER "${_saccade_gcc_root}/bin/gcc")
set(CMAKE_CXX_COMPILER "${_saccade_gcc_root}/bin/g++")
set(CMAKE_CUDA_HOST_COMPILER "${_saccade_gcc_root}/bin/g++")
