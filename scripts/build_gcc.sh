#!/usr/bin/env bash
# Build the pinned host compiler, GCC 16.2.0 (official release).
#
#   install    ~/.local/opt/gcc-16.2.0            (SACCADE_GCC_PREFIX)
#   links      ~/.local/bin/{gcc,g++}-16.2.0       (SACCADE_GCC_BINDIR)
#   downloads  ~/.cache/saccade-toolchain/src      (SACCADE_TOOLCHAIN_CACHE)
#
# This project:  cmake ... -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-gcc-16.2.0.cmake
# Other projects: CC=gcc-16.2.0 CXX=g++-16.2.0 CUDAHOSTCXX=g++-16.2.0
#
# The configure flags follow Arch Linux's gcc 16.2.1 package for everything
# that changes generated code or the default options (PIE, SSP, CET, threads,
# locale, hash style). Differences: C and C++ only, no multilib, plain
# bootstrap instead of bootstrap-lto, and GMP/MPFR/MPC/ISL built in-tree so the
# compiler does not depend on the system's copies.
set -euo pipefail

version=16.2.0
tarball_sha256=e6738e29597f733270731aa90600f37ffdc045079dfc27ec7e8192cc81085c3e
prefix=${SACCADE_GCC_PREFIX:-$HOME/.local/opt/gcc-$version}
bindir=${SACCADE_GCC_BINDIR:-$HOME/.local/bin}
cache=${SACCADE_TOOLCHAIN_CACHE:-$HOME/.cache/saccade-toolchain}
src=$cache/src
work=$cache/build/gcc-$version
jobs=${JOBS:-$(nproc)}
mirror=https://ftp.gnu.org/gnu

for path in "$prefix" "$bindir/gcc-$version" "$bindir/g++-$version"; do
  if [[ -e $path || -L $path ]]; then
    echo "error: $path already exists" >&2
    exit 1
  fi
done

mkdir -p "$src"
cd "$src"
fetch() { [[ -f $2 ]] || curl -fL --retry 3 -o "$2" "$1"; }
fetch "$mirror/gcc/gcc-$version/gcc-$version.tar.xz" "gcc-$version.tar.xz"
fetch "$mirror/gcc/gcc-$version/gcc-$version.tar.xz.sig" "gcc-$version.tar.xz.sig"
fetch "$mirror/gnu-keyring.gpg" gnu-keyring.gpg
gpgv --keyring "$src/gnu-keyring.gpg" "gcc-$version.tar.xz.sig" "gcc-$version.tar.xz"
echo "$tarball_sha256  gcc-$version.tar.xz" | sha256sum -c -

rm -rf "gcc-$version"
tar -xf "gcc-$version.tar.xz"
(cd "gcc-$version" && ./contrib/download_prerequisites)

rm -rf "$work"
mkdir -p "$work"
cd "$work"
"$src/gcc-$version/configure" \
  --prefix="$prefix" \
  --enable-languages=c,c++ \
  --disable-multilib \
  --enable-bootstrap \
  --with-gcc-major-version-only \
  --with-linker-hash-style=gnu \
  --with-system-zlib \
  --enable-cet=auto \
  --enable-checking=release \
  --enable-clocale=gnu \
  --enable-default-pie \
  --enable-default-ssp \
  --enable-gnu-indirect-function \
  --enable-gnu-unique-object \
  --enable-libstdcxx-backtrace \
  --enable-link-serialization=1 \
  --enable-linker-build-id \
  --enable-lto \
  --enable-plugin \
  --enable-shared \
  --enable-threads=posix \
  --disable-fixincludes \
  --disable-libssp \
  --disable-libstdcxx-pch \
  --disable-werror
make -j"$jobs"
make install-strip

mkdir -p "$bindir"
ln -s "$prefix/bin/gcc" "$bindir/gcc-$version"
ln -s "$prefix/bin/g++" "$bindir/g++-$version"
cd "$cache"
rm -rf "$work" "$src/gcc-$version"

"$bindir/g++-$version" --version | head -n 1
echo "installed: $prefix"
