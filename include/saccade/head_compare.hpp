#pragma once

#include "saccade/head_reference.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace saccade {

// Element-wise judgement of six raw head outputs against the reference under
// docs/HEAD_POLICY.md (s04-head-fp32/1): |gpu - ref| <= kLambda sigma + kFloor
// for every element, with the policy's report fields.
inline constexpr const char* kHeadPolicyVersion = "s04-head-fp32/1";

struct HeadOutputStats {
    std::size_t count = 0, failures = 0, rel_excluded = 0, nonfinite = 0;
    double max_abs = 0, max_rel = 0, max_ratio = 0, max_bound_ratio = 0;
    std::size_t worst = 0; // element index of the largest error / tolerance
    std::int64_t worst_c = 0, worst_h = 0, worst_w = 0;
    float worst_gpu = 0;
    double worst_ref = 0, worst_tol = 0;
};

struct HeadComparison {
    std::array<HeadOutputStats, 6> outputs;
    bool pass = true;
    // One line per output, tab separated, prefixed with `label`.
    std::string report(const std::string& label) const;
};

HeadComparison compare_head_outputs(const std::array<std::span<const float>, 6>& gpu, const HeadReferenceResult& reference);

} // namespace saccade
