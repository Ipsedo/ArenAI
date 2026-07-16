//
// Created by samuel on 01/07/2026.
//

#include <atomic>
#include <thread>
#include <vector>

#include <arenai_core/thread_pool.h>
#include <arenai_core_tests/tests_thread_limiter.h>

using namespace arenai;
using namespace arenai::core;

// ========================================================================
// Acquire / Release — basic behavior
// ========================================================================

TEST_F(ThreadLimiterTest, SingleAcquireRelease) {
    ThreadLimiter limiter(1);

    const uint64_t ticket = limiter.acquire();
    ASSERT_EQ(ticket, 0);

    limiter.release();
}

TEST_F(ThreadLimiterTest, SequentialAcquiresReturnIncreasingTickets) {
    ThreadLimiter limiter(1);

    const uint64_t t0 = limiter.acquire();
    limiter.release();

    const uint64_t t1 = limiter.acquire();
    limiter.release();

    const uint64_t t2 = limiter.acquire();
    limiter.release();

    ASSERT_EQ(t0, 0);
    ASSERT_EQ(t1, 1);
    ASSERT_EQ(t2, 2);
}

// ========================================================================
// Concurrency limiting
// ========================================================================

TEST_F(ThreadLimiterTest, ConcurrencyNeverExceedsK) {
    constexpr unsigned int k = 2;
    constexpr int num_threads = 8;
    constexpr int iterations_per_thread = 50;

    ThreadLimiter limiter(k);

    std::atomic<int> concurrent_count{0};
    std::atomic<int> max_concurrent{0};
    std::atomic<bool> violation{false};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&] {
            for (int j = 0; j < iterations_per_thread; j++) {
                limiter.acquire();

                const int current = concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;
                if (current > static_cast<int>(k)) violation.store(true, std::memory_order_relaxed);

                int prev_max = max_concurrent.load(std::memory_order_relaxed);
                while (prev_max < current
                       && !max_concurrent.compare_exchange_weak(
                           prev_max, current, std::memory_order_relaxed))
                    ;

                std::this_thread::yield();

                concurrent_count.fetch_sub(1, std::memory_order_relaxed);

                limiter.release();
            }
        });
    }

    for (auto &t: threads) t.join();

    ASSERT_FALSE(violation.load()) << "concurrent count exceeded k=" << k;
    ASSERT_GT(max_concurrent.load(), 0) << "at least one thread should have entered";
}

TEST_F(ThreadLimiterTest, ConcurrencyLimitOneActsAsMutex) {
    constexpr unsigned int k = 1;
    constexpr int num_threads = 4;
    constexpr int iterations_per_thread = 100;

    ThreadLimiter limiter(k);

    std::atomic<int> concurrent_count{0};
    std::atomic<bool> violation{false};

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&] {
            for (int j = 0; j < iterations_per_thread; j++) {
                limiter.acquire();

                if (const int current =
                        concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;
                    current > 1)
                    violation.store(true, std::memory_order_relaxed);

                std::this_thread::yield();

                concurrent_count.fetch_sub(1, std::memory_order_relaxed);

                limiter.release();
            }
        });
    }

    for (auto &t: threads) t.join();

    ASSERT_FALSE(violation.load()) << "k=1 should behave as a mutex";
}

// ========================================================================
// FIFO ordering
// ========================================================================

TEST_F(ThreadLimiterTest, FIFOOrdering) {
    constexpr unsigned int k = 1;
    constexpr int num_threads = 5;

    ThreadLimiter limiter(k);

    // Hold the limiter so all threads queue up
    limiter.acquire();

    std::vector<int> order;
    std::mutex order_mutex;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::atomic<int> ready_count{0};

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&, i] {
            ready_count.fetch_add(1, std::memory_order_release);
            limiter.acquire();

            {
                std::lock_guard lk(order_mutex);
                order.push_back(i);
            }

            limiter.release();
        });

        // Wait for the thread to be spawned and calling acquire
        while (ready_count.load(std::memory_order_acquire) <= i) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Release the initial hold — threads should proceed in FIFO order
    limiter.release();

    for (auto &t: threads) t.join();

    ASSERT_EQ(static_cast<int>(order.size()), num_threads);
    for (int i = 0; i < num_threads; i++) {
        ASSERT_EQ(order[i], i) << "thread " << i << " should have been served at position " << i;
    }
}

// ========================================================================
// Large K — all threads proceed immediately
// ========================================================================

TEST_F(ThreadLimiterTest, LargeKAllThreadsProceedImmediately) {
    constexpr int num_threads = 8;
    ThreadLimiter limiter(num_threads);

    std::atomic<int> concurrent_count{0};
    std::atomic<int> max_concurrent{0};
    std::barrier sync_point(num_threads);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&] {
            limiter.acquire();

            const int current = concurrent_count.fetch_add(1, std::memory_order_relaxed) + 1;
            int prev_max = max_concurrent.load(std::memory_order_relaxed);
            while (prev_max < current
                   && !max_concurrent.compare_exchange_weak(
                       prev_max, current, std::memory_order_relaxed))
                ;

            sync_point.arrive_and_wait();

            concurrent_count.fetch_sub(1, std::memory_order_relaxed);
            limiter.release();
        });
    }

    for (auto &t: threads) t.join();

    ASSERT_EQ(max_concurrent.load(), num_threads)
        << "all threads should run concurrently when k >= num_threads";
}
