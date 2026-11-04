#pragma once

#include <cstddef>
#include <exception>
#include <type_traits>
#include <utility>

namespace saccade {
// A free/destroy error may report earlier asynchronous work; the release outcome
// can be uncertain. Consume ownership before the attempt and never retry that
// handle. Retain the first exception while attempting all independent releases.
class CleanupReport {
public:
    template <typename Handle, typename Release>
    void release(Handle& owned, Release&& action) noexcept {
        static_assert(std::is_pointer_v<Handle> || std::is_integral_v<Handle>);
        if (owned == Handle{}) return;
        const auto handle = std::exchange(owned, Handle{});
        ++attempts_;
        try { std::forward<Release>(action)(handle); }
        catch (...) {
            ++failures_;
            if (!first_error_) first_error_ = std::current_exception();
        }
    }
    void throw_if_failed() const {
        if (first_error_) std::rethrow_exception(first_error_);
    }
    std::size_t attempts() const noexcept { return attempts_; }
    std::size_t failures() const noexcept { return failures_; }
private:
    std::exception_ptr first_error_;
    std::size_t attempts_ = 0, failures_ = 0;
};
} // namespace saccade
