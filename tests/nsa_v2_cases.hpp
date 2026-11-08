#pragma once
#include <array>
#include <cstdint>

// Independently rounded binary32 answers, including adjacent representable
// scores at both clamps (s0 and 1 - sqrt(30) * (1-s0)).
inline constexpr std::array<std::array<std::uint32_t, 2>, 11> nsa_v2_cases{{
    {0x00000000u, 0x41f00000u}, {0x3f000000u, 0x41f00000u},
    {0x3f6e353fu, 0x3f800000u}, {0x3f733333u, 0x3f800000u},
    {0x3f7d70a4u, 0x3f800000u}, {0x3f800000u, 0x3f800000u},
    {0x3f6e353eu, 0x3f80000eu}, {0x3f6e3540u, 0x3f800000u},
    {0x3f1e8c95u, 0x41f00000u}, {0x3f1e8c96u, 0x41effffeu},
    {0x3f1e8c97u, 0x41effffcu}
}};
