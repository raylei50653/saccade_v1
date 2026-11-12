#include "saccade/postprocess.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <vector>

extern "C" int f04_convert(const float* cls, const float* reg, unsigned width,
                          unsigned height, unsigned long long epoch,
                          unsigned long long frame, void* output) {
    try {
        constexpr auto profile = saccade::ModelProfile::Hr1280x768;
        constexpr auto spec = saccade::kHrModelProfile;
        std::array<std::vector<float>, 3> cls_levels, reg_levels;
        saccade::PostprocessRawView view;
        for (unsigned level = 0; level != 3; ++level) {
            const auto n = spec.level_positions(level), offset = spec.level_offset(level);
            cls_levels[level].resize(80 * n);
            reg_levels[level].resize(4 * n);
            for (unsigned c = 0; c != 80; ++c)
                std::copy_n(cls + c * spec.positions() + offset, n, cls_levels[level].data() + c * n);
            for (unsigned c = 0; c != 4; ++c)
                std::copy_n(reg + c * spec.positions() + offset, n, reg_levels[level].data() + c * n);
            view.cls[level] = cls_levels[level].data();
            view.reg[level] = reg_levels[level].data();
            view.cls_elements[level] = cls_levels[level].size();
            view.reg_elements[level] = reg_levels[level].size();
        }
        saccade::PostprocessRequest request;
        request.epoch = epoch;
        request.frame = frame;
        request.profile = profile;
        request.mapping = {static_cast<int>(width), static_cast<int>(height), 1280, 768,
                           1280.0 / width, 768.0 / height};
        const auto result = saccade::postprocess_reference(
            view, request, {0, saccade::kPostprocessProducerDone});
        std::memcpy(output, &result, sizeof(result));
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
