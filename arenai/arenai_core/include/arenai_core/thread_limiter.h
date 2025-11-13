//
// Created by samuel on 05/11/2025.
//

#ifndef ARENAI_TRAIN_HOST_THREAD_POOL_H
#define ARENAI_TRAIN_HOST_THREAD_POOL_H
#include <condition_variable>
#include <mutex>

class ThreadLimiter {
public:
    explicit ThreadLimiter(unsigned int k);

    uint64_t acquire();

    void release();

private:
    const unsigned int k_threads;
    std::mutex mutex;
    std::condition_variable condition_variable;
    uint64_t next_ticket = 0;
    uint64_t serving_ticket = 0;
};
#endif//ARENAI_TRAIN_HOST_THREAD_POOL_H
