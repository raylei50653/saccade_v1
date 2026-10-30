#include "saccade/head_compare.hpp"

#include <bit>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace saccade {

HeadComparison compare_head_outputs(const std::array<std::span<const float>, 6>& gpu, const HeadReferenceResult& reference) {
    using namespace head_policy;
    HeadComparison result;
    for (std::size_t k = 0; k < 6; ++k) {
        const auto& ref = reference.outputs[k];
        const auto& got = gpu[k];
        if (got.size() != ref.value.size()) throw std::invalid_argument(std::string("head comparison: element count of ") + kHeadOutputNames[k]);
        auto& st = result.outputs[k];
        st.count = got.size();
        for (std::size_t i = 0; i < got.size(); ++i) {
            const double g = got[i], err = std::fabs(g - ref.value[i]), tol = kLambda * ref.sigma[i] + kFloor;
            if (!std::isfinite(g)) ++st.nonfinite;
            const double ratio = std::isfinite(err) ? err / tol : std::numeric_limits<double>::infinity();
            if (!(err <= tol)) ++st.failures;
            if (!(ratio <= st.max_ratio)) {
                st.max_ratio = ratio;
                st.worst = i;
            }
            if (std::isfinite(err)) st.max_abs = std::max(st.max_abs, err);
            // Relative error only where the value is not lost in its own tolerance.
            if (std::fabs(ref.value[i]) >= tol) st.max_rel = std::max(st.max_rel, err / std::fabs(ref.value[i]));
            else ++st.rel_excluded;
            st.max_bound_ratio = std::max(st.max_bound_ratio, err / (2 * ref.bound[i]));
        }
        const auto& s = ref.shape; // (1, C, H, W)
        st.worst_c = static_cast<std::int64_t>(st.worst) / (s[2] * s[3]);
        st.worst_h = static_cast<std::int64_t>(st.worst) / s[3] % s[2];
        st.worst_w = static_cast<std::int64_t>(st.worst) % s[3];
        if (st.count) {
            st.worst_gpu = got[st.worst];
            st.worst_ref = ref.value[st.worst];
            st.worst_tol = kLambda * ref.sigma[st.worst] + kFloor;
        }
        result.pass = result.pass && st.failures == 0 && st.nonfinite == 0;
    }
    return result;
}

std::string HeadComparison::report(const std::string& label) const {
    std::ostringstream out;
    for (std::size_t k = 0; k < 6; ++k) {
        const auto& st = outputs[k];
        out << "head_compare\t" << label << '\t' << kHeadOutputNames[k] << "\tpolicy=" << kHeadPolicyVersion << "\tcount=" << st.count
            << std::setprecision(3) << "\tmax_abs=" << st.max_abs << "\tmax_rel=" << st.max_rel << "\trel_excluded=" << st.rel_excluded
            << "\tmax_err_over_tol=" << st.max_ratio << "\tfailures=" << st.failures << "\tnonfinite=" << st.nonfinite << "\tworst=(c=" << st.worst_c
            << ",h=" << st.worst_h << ",w=" << st.worst_w << ")" << std::setprecision(9) << "\tworst_gpu=" << st.worst_gpu << "(0x" << std::hex
            << std::setw(8) << std::setfill('0') << std::bit_cast<std::uint32_t>(st.worst_gpu) << std::dec << std::setfill(' ') << ")"
            << std::setprecision(17) << "\tworst_ref=" << st.worst_ref << std::setprecision(3) << "\tworst_tol=" << st.worst_tol
            << "\tmax_err_over_2bound=" << st.max_bound_ratio << '\t' << (st.failures || st.nonfinite ? "FAIL" : "PASS") << '\n';
    }
    return out.str();
}

} // namespace saccade
