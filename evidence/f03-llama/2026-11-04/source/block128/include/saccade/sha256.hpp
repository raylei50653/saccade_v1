#pragma once

#include <string>
#include <string_view>

namespace saccade {
// FIPS 180-4, lowercase hex. Byte-preserving, independent of configuration encoding.
std::string sha256_hex(std::string_view bytes);
} // namespace saccade
