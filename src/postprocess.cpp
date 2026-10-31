#include "saccade/postprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace saccade {
namespace {
struct Ranked {
    DetectionCandidate row{}; // model coordinates until final mapping
};

DetectionCandidates empty_result(const PostprocessRequest& request, std::uint32_t flags = 0) {
    DetectionCandidates output{};
    output.header.epoch = request.epoch;
    output.header.frame = request.frame;
    output.header.schema = kPostprocessSchema;
    output.header.capacity = request.capacity;
    output.header.flags = flags;
    output.header.done = kPostprocessDone;
    return output;
}

float sigmoid(float value) {
    const float e = std::exp(-std::abs(value));
    const float denominator = 1.0f + e;
    return (value >= 0.0f ? 1.0f : e) / denominator;
}

bool area(const DetectionCandidate& box, float& result) {
    const float width = box.x2 - box.x1;
    const float height = box.y2 - box.y1;
    result = width * height;
    return std::isfinite(width) && std::isfinite(height) && std::isfinite(result);
}

// Deliberately scalar, retaining all rounded intermediates. The CUDA kernels
// implement this contract independently with explicit FP32 RN intrinsics.
bool overlap(const DetectionCandidate& a, const DetectionCandidate& b, bool& suppressed) {
    float area_a = 0.0f, area_b = 0.0f;
    if (!area(a, area_a) || !area(b, area_b)) return false;
    const float dx = std::min(a.x2, b.x2) - std::max(a.x1, b.x1);
    const float dy = std::min(a.y2, b.y2) - std::max(a.y1, b.y1);
    const float width = std::max(0.0f, dx), height = std::max(0.0f, dy);
    const float intersection = width * height;
    const float area_sum = area_a + area_b;
    const float union_area = area_sum - intersection;
    if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(intersection) ||
        !std::isfinite(area_sum) || !std::isfinite(union_area) || union_area <= 0.0f)
        return false;
    const float iou = intersection / union_area;
    if (!std::isfinite(iou)) return false;
    suppressed = iou > kPostprocessNmsIou;
    return true;
}
} // namespace

void validate_postprocess_arguments(const PostprocessRawView& raw, const PostprocessRequest& request) {
    if (request.capacity > kPostprocessCapacity)
        throw std::invalid_argument("F04 capacity exceeds fixed top-300 profile");
    const auto& mapping = request.mapping;
    if (mapping.original_width < 1 || mapping.original_width > 65535 ||
        mapping.original_height < 1 || mapping.original_height > 65535 ||
        mapping.model_width != 640 || mapping.model_height != 640 ||
        mapping.scale_x != 640.0 / mapping.original_width ||
        mapping.scale_y != 640.0 / mapping.original_height)
        throw std::invalid_argument("F04 requires a finite, exact 640 stretch ImageMapping");
    for (std::size_t level = 0; level < kPostprocessSizes.size(); ++level) {
        const auto spatial = std::size_t{kPostprocessSizes[level]} * kPostprocessSizes[level];
        if (raw.cls[level] == nullptr || raw.reg[level] == nullptr ||
            raw.cls_elements[level] != kPostprocessClasses * spatial ||
            raw.reg_elements[level] != 4 * spatial)
            throw std::invalid_argument("F04 requires all six fixed-shape raw tensors");
        if (reinterpret_cast<std::uintptr_t>(raw.cls[level]) % alignof(float) != 0 ||
            reinterpret_cast<std::uintptr_t>(raw.reg[level]) % alignof(float) != 0)
            throw std::invalid_argument("F04 requires aligned FP32 raw tensors");
    }
}

DetectionCandidates postprocess_reference(const PostprocessRawView& raw,
                                           const PostprocessRequest& request,
                                           const PostprocessProducerStatus& producer) {
    validate_postprocess_arguments(raw, request);
    std::uint32_t failure = 0;
    if (producer.flags != 0) failure |= kPostprocessUpstream;
    if (producer.done != kPostprocessProducerDone) failure |= kPostprocessProtocol;
    if (failure != 0) return empty_result(request, failure);

    // All raw elements are checked, including entries that filtering would
    // discard. No NaN may silently disappear through max, top-k or NMS.
    for (std::size_t level = 0; level < kPostprocessSizes.size(); ++level) {
        for (std::size_t i = 0; i < raw.cls_elements[level]; ++i)
            if (!std::isfinite(raw.cls[level][i])) return empty_result(request, kPostprocessNumeric);
        for (std::size_t i = 0; i < raw.reg_elements[level]; ++i)
            if (!std::isfinite(raw.reg[level][i])) return empty_result(request, kPostprocessNumeric);
    }

    std::vector<Ranked> ranked;
    ranked.reserve(kPostprocessPositions);
    std::uint32_t id = 0;
    for (std::size_t level = 0; level < kPostprocessSizes.size(); ++level) {
        const auto size = kPostprocessSizes[level];
        const auto spatial = size * size;
        const auto* cls = raw.cls[level];
        const auto* reg = raw.reg[level];
        const auto stride = static_cast<float>(kPostprocessStrides[level]);
        for (std::uint32_t location = 0; location < spatial; ++location, ++id) {
            DetectionCandidate row{};
            row.score = sigmoid(cls[location]);
            for (std::uint32_t c = 1; c < kPostprocessClasses; ++c) {
                const float score = sigmoid(cls[c * spatial + location]);
                if (score > row.score) { row.score = score; row.class_id = c; }
            }
            row.candidate_id = id;
            const float x = static_cast<float>(location % size) + 0.5f;
            const float y = static_cast<float>(location / size) + 0.5f;
            const float left = x - reg[location];
            const float top = y - reg[spatial + location];
            const float right = x + reg[2 * spatial + location];
            const float bottom = y + reg[3 * spatial + location];
            row.x1 = left * stride; row.y1 = top * stride;
            row.x2 = right * stride; row.y2 = bottom * stride;
            if (!std::isfinite(row.x1) || !std::isfinite(row.y1) ||
                !std::isfinite(row.x2) || !std::isfinite(row.y2))
                return empty_result(request, kPostprocessNumeric);
            if (row.score >= kPostprocessScoreFloor) ranked.push_back({row});
        }
    }

    DetectionCandidates output = empty_result(request);
    output.header.low_score = kPostprocessPositions - static_cast<std::uint32_t>(ranked.size());
    std::sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
        return a.row.score > b.row.score ||
            (a.row.score == b.row.score && a.row.candidate_id < b.row.candidate_id);
    });
    if (ranked.size() > kPostprocessCapacity) {
        output.header.topk_dropped = static_cast<std::uint32_t>(ranked.size()) - kPostprocessCapacity;
        ranked.resize(kPostprocessCapacity);
    }
    std::vector<DetectionCandidate> persons;
    persons.reserve(kPostprocessCapacity);
    for (const auto& item : ranked) {
        if (item.row.class_id != 0) { ++output.header.class_dropped; continue; }
        if (item.row.x2 <= item.row.x1 || item.row.y2 <= item.row.y1) {
            ++output.header.degenerate; continue;
        }
        persons.push_back(item.row);
    }
    if (persons.size() > request.capacity) return empty_result(request, kPostprocessOverflow);
    output.header.count = static_cast<std::uint32_t>(persons.size());
    for (std::size_t i = 0; i < persons.size(); ++i) {
        float box_area = 0.0f;
        if (!area(persons[i], box_area) || box_area <= 0.0f)
            return empty_result(request, kPostprocessNumeric);
        bool suppressed = false;
        for (std::size_t j = 0; j < i; ++j) {
            if (persons[j].flags != kDetectionKept) continue;
            bool pair_suppressed = false;
            if (!overlap(persons[j], persons[i], pair_suppressed))
                return empty_result(request, kPostprocessNumeric);
            if (pair_suppressed) { suppressed = true; break; }
        }
        persons[i].flags = suppressed ? kDetectionSuppressed : kDetectionKept;
        if (suppressed) ++output.header.suppressed; else ++output.header.kept;
    }

    const float scale_x = static_cast<float>(request.mapping.scale_x);
    const float scale_y = static_cast<float>(request.mapping.scale_y);
    for (std::size_t i = 0; i < persons.size(); ++i) {
        auto row = persons[i];
        row.x1 = row.x1 / scale_x; row.x2 = row.x2 / scale_x;
        row.y1 = row.y1 / scale_y; row.y2 = row.y2 / scale_y;
        if (!std::isfinite(row.x1) || !std::isfinite(row.y1) ||
            !std::isfinite(row.x2) || !std::isfinite(row.y2))
            return empty_result(request, kPostprocessNumeric);
        output.rows[i] = row;
    }
    return output;
}

} // namespace saccade
