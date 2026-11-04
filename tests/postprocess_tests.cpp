#include "saccade/postprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace saccade;
std::size_t checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
template<class F> void rejected(F&& call) {
    bool caught = false;
    try { call(); } catch (const std::invalid_argument&) { caught = true; }
    check(caught, "F04 invalid host argument accepted");
}
struct Fixture {
    std::array<std::vector<float>, 3> cls, reg;
    PostprocessRequest request{};
    PostprocessProducerStatus producer{0, kPostprocessProducerDone};
    explicit Fixture(ModelProfile profile = kDefaultModelProfile) {
        request.profile = profile;
        const auto& spec = model_profile(profile);
        request.mapping = {static_cast<int>(spec.width), static_cast<int>(spec.height),
                           static_cast<int>(spec.width), static_cast<int>(spec.height), 1.0, 1.0};
        request.epoch = 0xfedcba9876543210ULL;
        request.frame = 0x123456789abcdef0ULL;
        for (std::size_t level = 0; level < cls.size(); ++level) {
            const auto spatial = model_profile(request.profile).level_positions(level);
            cls[level].assign(kPostprocessClasses * spatial, -100.0f);
            reg[level].assign(4 * spatial, 0.5f);
        }
    }
    PostprocessRawView raw() const {
        PostprocessRawView view;
        for (std::size_t level = 0; level < cls.size(); ++level) {
            view.cls[level] = cls[level].data(); view.reg[level] = reg[level].data();
            view.cls_elements[level] = cls[level].size(); view.reg_elements[level] = reg[level].size();
        }
        return view;
    }
    std::pair<std::size_t, std::uint32_t> location(std::uint32_t id) const {
        const auto& spec = model_profile(request.profile);
        for (std::size_t level = 0; level < 3; ++level) {
            if (id < spec.level_positions(level)) return {level, id};
            id -= spec.level_positions(level);
        }
        throw std::runtime_error("invalid fixture ID");
    }
    void score(std::uint32_t id, float logit, std::uint32_t class_id = 0) {
        const auto [level, pos] = location(id);
        const auto spatial = model_profile(request.profile).level_positions(level);
        cls[level][class_id * spatial + pos] = logit;
    }
    void box(std::uint32_t id, float x1, float y1, float x2, float y2) {
        const auto [level, pos] = location(id);
        const auto size = static_cast<std::uint32_t>(model_profile(request.profile).level_width(level));
        const auto spatial = model_profile(request.profile).level_positions(level);
        const float stride = static_cast<float>(kPostprocessStrides[level]);
        const float x = static_cast<float>(pos % size) + 0.5f;
        const float y = static_cast<float>(pos / size) + 0.5f;
        reg[level][pos] = x - x1 / stride;
        reg[level][spatial + pos] = y - y1 / stride;
        reg[level][2 * spatial + pos] = x2 / stride - x;
        reg[level][3 * spatial + pos] = y2 / stride - y;
    }
    DetectionCandidates run() const { return postprocess_reference(raw(), request, producer); }
};

void padding(const DetectionCandidates& output) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(output.rows.data() + output.header.count);
    const auto length = (kPostprocessCapacity - output.header.count) * sizeof(DetectionCandidate);
    check(std::all_of(bytes, bytes + length, [](unsigned char x) { return x == 0; }), "F04 unused rows not zero");
    check(output.header.reserved == 0, "F04 reserved header not zero");
}
void failed(const DetectionCandidates& output, const PostprocessRequest& request, std::uint32_t flag) {
    check(output.header.epoch == request.epoch && output.header.frame == request.frame, "F04 failure lost identity");
    check(output.header.schema == kPostprocessSchema && output.header.capacity == request.capacity,
          "F04 failure lost schema/capacity");
    check(output.header.flags == flag && output.header.done == kPostprocessDone, "F04 incorrect failure flags/completion");
    check(output.header.count == 0 && output.header.kept == 0 && output.header.suppressed == 0 &&
          output.header.low_score == 0 && output.header.topk_dropped == 0 && output.header.class_dropped == 0 &&
          output.header.degenerate == 0, "F04 failure exposed partial output/statistics");
    padding(output);
}
void basic() {
    Fixture fixture;
    fixture.request.capacity = 0;
    auto output = fixture.run();
    check(output.header.flags == 0 && output.header.done == kPostprocessDone && output.header.count == 0,
          "F04 empty frame is not a valid empty output");
    check(output.header.low_score == 8400 && output.header.capacity == 0, "F04 empty-frame accounting");
    padding(output);
    fixture.request.capacity = 300;
    fixture.request.mapping = {1280, 320, 640, 640, 0.5, 2.0};
    fixture.score(0, 0.0f);
    fixture.box(0, -4.0f, -4.0f, 12.0f, 12.0f);
    output = fixture.run();
    check(output.header.count == 1 && output.header.kept == 1 && output.header.suppressed == 0 &&
          output.header.low_score == 8399, "F04 single-candidate accounting");
    check(output.rows[0].x1 == -8.0f && output.rows[0].x2 == 24.0f &&
          output.rows[0].y1 == -2.0f && output.rows[0].y2 == 6.0f, "F04 mapping or unintended clipping");
    check(output.rows[0].score == 0.5f && output.rows[0].candidate_id == 0 && output.rows[0].class_id == 0 &&
          output.rows[0].flags == kDetectionKept, "F04 single-candidate fields");
    padding(output);
    const auto repeat = fixture.run();
    check(std::memcmp(&output, &repeat, sizeof(output)) == 0, "F04 reference repeat differs");
    fixture.request.mapping = {1920, 1080, 640, 640, 640.0 / 1920.0, 640.0 / 1080.0};
    fixture.box(0, -3.0f, 7.0f, 21.0f, 39.0f);
    output = fixture.run();
    check(output.rows[0].x1 == -3.0f / static_cast<float>(640.0 / 1920.0) &&
          output.rows[0].y1 == 7.0f / static_cast<float>(640.0 / 1080.0) &&
          output.rows[0].x2 == 21.0f / static_cast<float>(640.0 / 1920.0) &&
          output.rows[0].y2 == 39.0f / static_cast<float>(640.0 / 1080.0),
          "F04 original mapping must divide by rounded forward scale");
}

void classes_and_order() {
    Fixture fixture;
    fixture.score(0, 0.0f); fixture.score(0, 0.0f, 7);
    fixture.score(6400, 100.0f); fixture.score(6400, 101.0f, 7);
    fixture.score(8000, 0.0f);
    fixture.score(1, 100.0f, 3); fixture.score(1, 101.0f, 7);
    auto output = fixture.run();
    check(output.header.count == 3 && output.header.class_dropped == 1, "F04 class filtering or class tie");
    check(output.rows[0].candidate_id == 6400 && output.rows[0].score == 1.0f && output.rows[0].class_id == 0,
          "F04 must tie on rounded sigmoid, not raw logits");
    check(output.rows[1].candidate_id == 0 && output.rows[2].candidate_id == 8000,
          "F04 stable score/ID order across levels");
    padding(output);

    Fixture saturated;
    for (std::uint32_t id = 0; id < 301; ++id) saturated.score(id, id == 300 ? 101.0f : 100.0f);
    output = saturated.run();
    check(output.header.count == 300 && output.header.topk_dropped == 1, "F04 fixed top300 selection");
    for (std::uint32_t i = 0; i < 300; ++i)
        check(output.rows[i].candidate_id == i && output.rows[i].score == 1.0f, "F04 top300 sigmoid tie ordering");
    padding(output);
}

void topk_before_filters() {
    Fixture fixture;
    for (std::uint32_t i = 0; i < 300; ++i) fixture.score(i, 0.0f, 1);
    fixture.score(300, -0.04f);
    auto output = fixture.run();
    check(output.header.count == 0 && output.header.class_dropped == 300 && output.header.topk_dropped == 1 &&
          output.header.low_score == 8099, "F04 person-only topk or refilling after class filtering");
    padding(output);
    for (std::uint32_t i = 0; i < 300; ++i) {
        fixture.score(i, 0.1f);
        fixture.box(i, 0.0f, 0.0f, 0.0f, 0.0f);
    }
    output = fixture.run();
    check(output.header.count == 0 && output.header.degenerate == 300 && output.header.topk_dropped == 1 &&
          output.header.class_dropped == 0, "F04 refilling after degenerate filtering");
    padding(output);
}

void nms() {
    Fixture fixture;
    fixture.score(0, 0.0f); fixture.score(1, 0.0f); fixture.score(2, 0.0f);
    // A suppresses B, B overlaps C, A does not overlap C sufficiently.
    fixture.box(0, 0.0f, 0.0f, 16.0f, 16.0f);
    fixture.box(1, 4.0f, 0.0f, 20.0f, 16.0f);
    fixture.box(2, 8.0f, 0.0f, 24.0f, 16.0f);
    auto output = fixture.run();
    check(output.header.count == 3 && output.header.kept == 2 && output.header.suppressed == 1,
          "F04 greedy NMS accounting");
    check(output.rows[0].flags == kDetectionKept && output.rows[1].flags == kDetectionSuppressed &&
          output.rows[2].flags == kDetectionKept, "F04 suppressed candidate incorrectly suppresses another");
    check(output.rows[1].candidate_id == 1 && output.rows[1].score == 0.5f, "F04 suppressed pool lost source information");
    padding(output);

    Fixture boundary;
    boundary.score(0, 0.0f); boundary.score(1, 0.0f);
    boundary.box(0, 0.0f, 0.0f, 16.0f, 16.0f);
    boundary.box(1, 0.0f, 0.0f, 8.0f, 16.0f); // IoU exactly 1/2
    output = boundary.run();
    check(output.header.kept == 2 && output.header.suppressed == 0, "F04 IoU==0.5 must survive");
    boundary.box(1, 0.0f, 0.0f, 9.0f, 16.0f);
    check(boundary.run().rows[1].flags == kDetectionSuppressed, "F04 IoU>0.5 must suppress");
    boundary.box(1, 0.0f, 0.0f, 7.0f, 16.0f);
    check(boundary.run().rows[1].flags == kDetectionKept, "F04 IoU<0.5 must survive");
}

void nms_parallel_cases() {
    // W*H = 2^125: sums below 8 units are finite, sums above 8 overflow.
    // Power-of-two coordinates survive raw ltrb encoding and decoding exactly.
    const float w = std::ldexp(1.0f, 62), h = std::ldexp(1.0f, 63);
    const auto near = [&](Fixture& f, unsigned id, float width) {
        f.box(id, 0, 0, width * w, h);
    };
    const auto far = [&](Fixture& f, unsigned id, float width) {
        f.box(id, 8 * w, 0, (8 + width) * w, h);
    };
    const auto three = [] {
        Fixture f;
        f.score(0, 100); f.score(1, 0); f.score(2, 0);
        return f;
    };
    const auto success = [](const Fixture& f, const char* name, unsigned suppressed) {
        const auto out = f.run();
        check(out.header.flags == 0 && out.header.done == kPostprocessDone &&
              out.header.count == 3 && out.header.kept == 2 && out.header.suppressed == 1 &&
              out.header.low_score == 8397 && out.header.topk_dropped == 0 &&
              out.header.class_dropped == 0 && out.header.degenerate == 0, name);
        for (unsigned i = 0; i < 3; ++i)
            check(out.rows[i].candidate_id == i && out.rows[i].class_id == 0 &&
                  out.rows[i].score == (i == 0 ? 1.0f : 0.5f) &&
                  out.rows[i].flags == (i == suppressed ? kDetectionSuppressed : kDetectionKept), name);
        padding(out);
        std::cout << "case=" << name << " PASS\n";
    };
    auto f = three();
    near(f, 0, 3); near(f, 1, 4); far(f, 2, 4.5f);
    // A/B IoU=3/4, C/A IoU=0; only the unvisited C/B sum (8.5 units) overflows.
    success(f, "nms-uncompared-suppressed-pair", 1);

    f = three();
    near(f, 0, 2.5f); far(f, 1, 5); near(f, 2, 3.5f);
    // j1/j2 sum=7.5; i/j1 IoU=5/7 decides i before the i/j2 sum=8.5.
    success(f, "nms-uncompared-after-decision", 2);

    f = three();
    far(f, 0, 4.5f); near(f, 1, 3); near(f, 2, 4);
    // Both earlier rows survive (sum=7.5, IoU=0); i/j1 fails before i/j2 IoU=3/4.
    failed(f.run(), f.request, kPostprocessNumeric);
    std::cout << "case=nms-compared-failure-before-over PASS\n";

    f = three();
    near(f, 0, 3); far(f, 1, 4.5f); near(f, 2, 4);
    success(f, "nms-compared-over-before-failure", 2);

    Fixture words;
    for (unsigned i = 0; i < 300; ++i) {
        words.score(i, 0);
        words.box(i, 64.0f * i, 0, 64.0f * i + 16, 16);
    }
    // 63 -> 64 -> 65 is a chain: 64 is suppressed, so 65 survives (IoU=1/3 with 63).
    words.box(64, 4036, 0, 4052, 16);
    words.box(65, 4040, 0, 4056, 16);
    for (unsigned j : {127u, 191u, 255u, 257u})
        words.box(j + 1, 64.0f * j, 0, 64.0f * j + 16, 16);
    words.box(299, 0, 0, 16, 16);
    const auto out = words.run();
    check(out.header.flags == 0 && out.header.done == kPostprocessDone && out.header.count == 300 &&
          out.header.kept == 294 && out.header.suppressed == 6 && out.header.low_score == 8100 &&
          out.header.topk_dropped == 0 && out.header.class_dropped == 0 && out.header.degenerate == 0,
          "nms-word-boundaries: complete pool");
    for (unsigned i = 0; i < 300; ++i) {
        const bool suppressed = i == 64 || i == 128 || i == 192 || i == 256 || i == 258 || i == 299;
        check(out.rows[i].candidate_id == i && out.rows[i].score == 0.5f && out.rows[i].class_id == 0 &&
              out.rows[i].flags == (suppressed ? kDetectionSuppressed : kDetectionKept),
              "nms-word-boundaries: hand-computed flags and order");
    }
    padding(out);
    std::cout << "case=nms-word-boundaries PASS\n";
}

void threshold() {
    // Identify neighboring FP32 logits spanning the FP32 score floor. This is
    // independent of the reference's candidate selection and tests >= exactly.
    const auto score = [](float value) {
        const float e = std::exp(-std::abs(value));
        const float denominator = 1.0f + e;
        return (value >= 0.0f ? 1.0f : e) / denominator;
    };
    float below = std::log(kPostprocessScoreFloor / (1.0f - kPostprocessScoreFloor));
    while (score(below) >= kPostprocessScoreFloor) below = std::nextafter(below, -std::numeric_limits<float>::infinity());
    float above = std::nextafter(below, std::numeric_limits<float>::infinity());
    check(score(below) < kPostprocessScoreFloor && score(above) >= kPostprocessScoreFloor,
          "F04 threshold neighbors do not bracket");
    Fixture fixture; fixture.score(0, below); fixture.score(1, above);
    const auto output = fixture.run();
    check(output.header.count == 1 && output.rows[0].candidate_id == 1 && output.header.low_score == 8399,
          "F04 score-floor neighbor selection");
}

void failures() {
    Fixture fixture; fixture.score(0, 0.0f);
    fixture.request.capacity = 0;
    failed(fixture.run(), fixture.request, kPostprocessOverflow);
    fixture.request.capacity = 300;
    for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity()}) {
        fixture.cls[2].back() = bad;
        failed(fixture.run(), fixture.request, kPostprocessNumeric);
        fixture.cls[2].back() = -100.0f;
        fixture.reg[2].back() = bad;
        failed(fixture.run(), fixture.request, kPostprocessNumeric);
        fixture.reg[2].back() = 0.5f;
    }
    // Decode overflow at a low-score, top-k-discarded location still fails.
    fixture.reg[2].back() = std::numeric_limits<float>::max();
    failed(fixture.run(), fixture.request, kPostprocessNumeric);
    fixture.reg[2].back() = 0.5f;
    fixture.box(0, -1e20f, -1e20f, 1e20f, 1e20f);
    failed(fixture.run(), fixture.request, kPostprocessNumeric); // finite coordinates, area overflows
    fixture.request.capacity = 0;
    failed(fixture.run(), fixture.request, kPostprocessOverflow); // capacity check precedes NMS
    fixture.request.capacity = 300;
    fixture.producer.flags = 0x80000000u;
    failed(fixture.run(), fixture.request, kPostprocessUpstream);
    fixture.producer.done = 0;
    failed(fixture.run(), fixture.request, kPostprocessUpstream | kPostprocessProtocol);
    fixture.producer.flags = 0;
    failed(fixture.run(), fixture.request, kPostprocessProtocol);
    fixture.producer.done = kPostprocessProducerDone;
    fixture.box(0, 0.0f, 0.0f, 8.0f, 8.0f);
    check(fixture.run().header.flags == 0, "F04 failure contaminates a subsequent call");

    Fixture union_overflow;
    union_overflow.score(0, 0.0f); union_overflow.score(1, 0.0f);
    union_overflow.box(0, 0.0f, 0.0f, 1.4e19f, 1.4e19f);
    union_overflow.box(1, 1.4e19f, 0.0f, 2.8e19f, 1.4e19f);
    failed(union_overflow.run(), union_overflow.request, kPostprocessNumeric);

    Fixture negative_sides;
    negative_sides.score(0, 0.0f);
    // Raw distances may be negative. Classify the decoded box rather than
    // clamping the raw distances, which would create a different observation.
    const float adjacent = std::nextafter(0.5f, 0.0f);
    negative_sides.reg[0][0] = adjacent;
    negative_sides.reg[0][6400] = adjacent;
    negative_sides.reg[0][12800] = -0.5f;
    negative_sides.reg[0][19200] = -0.5f;
    const auto degenerate = negative_sides.run();
    check(degenerate.header.flags == 0 && degenerate.header.count == 0 && degenerate.header.degenerate == 1,
          "F04 negative raw sides must use decoded degeneracy rather than raw sign");
}

void arguments() {
    Fixture fixture;
    auto raw = fixture.raw();
    auto request = fixture.request;
    request.capacity = 301;
    rejected([&] { (void)postprocess_reference(raw, request, fixture.producer); });
    for (const auto mapping : {ImageMapping{0, 640, 640, 640, 1, 1},
                               ImageMapping{640, 65536, 640, 640, 1, 1},
                               ImageMapping{640, 640, 320, 640, 1, 1},
                               ImageMapping{640, 640, 640, 640, 0.5, 1},
                               ImageMapping{640, 640, 640, 640, std::numeric_limits<double>::quiet_NaN(), 1}}) {
        request = fixture.request; request.mapping = mapping;
        rejected([&] { validate_postprocess_arguments(raw, request); });
    }
    request = fixture.request;
    for (std::size_t level = 0; level < 3; ++level) {
        raw = fixture.raw(); raw.cls[level] = nullptr;
        rejected([&] { validate_postprocess_arguments(raw, request); });
        raw = fixture.raw(); raw.reg[level] = nullptr;
        rejected([&] { validate_postprocess_arguments(raw, request); });
        raw = fixture.raw(); --raw.cls_elements[level];
        rejected([&] { validate_postprocess_arguments(raw, request); });
        raw = fixture.raw(); ++raw.reg_elements[level];
        rejected([&] { validate_postprocess_arguments(raw, request); });
        raw = fixture.raw(); raw.cls[level] = reinterpret_cast<const float*>(reinterpret_cast<const char*>(raw.cls[level]) + 1);
        rejected([&] { validate_postprocess_arguments(raw, request); });
    }
}
void hr_profile() {
    Fixture f(ModelProfile::Hr1280x768);
    check(kHrModelProfile.positions() == 20160 && kHrModelProfile.input_elements() == 2949120, "HR extents");
    check(kHrModelProfile.scan_length(0) == 960 && kHrModelProfile.scan_length(1) == 240 &&
          kHrModelProfile.scan_length(2) == 60, "HR scan lengths");
    rejected([] { (void)parse_model_profile("custom"); });
    check(parse_model_profile("hr-1280x768") == ModelProfile::Hr1280x768 &&
          parse_model_profile("t01-640x640") == kDefaultModelProfile, "profile names");
    constexpr std::array<std::uint32_t, 6> ids{0, 15359, 15360, 19199, 19200, 20159};
    for (auto id : ids) f.score(id, 0);
    auto result = f.run();
    check(result.header.count == 6 && result.header.low_score == 20154, "HR level endpoints");
    for (std::size_t i = 0; i < ids.size(); ++i) {
        check(result.rows[i].candidate_id == ids[i] && result.rows[i].score == 0.5f, "HR stable level IDs");
        const auto stride = static_cast<float>(8u << (i / 2));
        check(result.rows[i].x1 == (i % 2 ? 1280 - stride : 0) &&
              result.rows[i].y1 == (i % 2 ? 768 - stride : 0) &&
              result.rows[i].x2 == (i % 2 ? 1280 : stride) &&
              result.rows[i].y2 == (i % 2 ? 768 : stride), "HR rectangular anchor coordinates");
    }
    auto wrong = f.request;
    wrong.profile = kDefaultModelProfile;
    rejected([&] { validate_postprocess_arguments(f.raw(), wrong); });
    wrong = f.request; wrong.mapping.model_height = 640;
    rejected([&] { validate_postprocess_arguments(f.raw(), wrong); });
    Fixture dense(ModelProfile::Hr1280x768);
    for (std::uint32_t id = 0; id < 20160; ++id) dense.score(id, id >= 19860 ? 1.0f : 0.0f);
    result = dense.run();
    check(result.header.count == 300 && result.header.topk_dropped == 19860, "HR dense top300");
    for (std::uint32_t i = 0; i < 300; ++i)
        check(result.rows[i].candidate_id == 19860 + i, "HR candidate IDs above 14 bits");
    padding(result);
}
} // namespace

int main() {
    try {
        basic(); classes_and_order(); topk_before_filters(); nms(); nms_parallel_cases(); threshold(); failures(); arguments(); hr_profile();
        std::cout << "postprocess_reference: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "postprocess_reference: " << error.what() << '\n';
        return 1;
    }
}
