//
// Created by samuel on 05/11/2025.
//

#include <arenai_core/thread_limiter.h>

ThreadLimiter::ThreadLimiter(const unsigned int k) : k_threads(k) {}

uint64_t ThreadLimiter::acquire() {
    std::unique_lock lk(mutex);
    const uint64_t my = next_ticket++;
    condition_variable.wait(
        lk, [&] { return my < serving_ticket + static_cast<uint64_t>(k_threads); });
    return my;
}

void ThreadLimiter::release() {
    std::lock_guard lk(mutex);
    ++serving_ticket;
    condition_variable.notify_all();
}
