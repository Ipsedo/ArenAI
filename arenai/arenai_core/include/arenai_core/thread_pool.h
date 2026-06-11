//
// Created by samuel on 10/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_THREAD_POOL_H
#define ARENAI_TRAIN_HOST_THREAD_POOL_H
#include <barrier>
#include <condition_variable>
#include <memory>
#include <vector>

#include <arenai_utils/double_buffer.h>

#include "./enemy_tank_factory.h"

class VisionDoubleBuffer : public DoubleBuffer<image<uint8_t>> {
public:
    VisionDoubleBuffer(int height, int width);

private:
    static image<uint8_t> black_image(int height, int width);
};

class ModelMatricesDoubleBuffer
    : public DoubleBuffer<std::vector<std::tuple<std::string, glm::mat4>>> {
public:
    ModelMatricesDoubleBuffer();
};

/*
 * Threads
 */

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

class EnemyVisionThreadPool {
public:
    EnemyVisionThreadPool(
        int num_threads, const std::vector<std::shared_ptr<EnemyTankFactory>> &enemy_tank_factories,
        bool thread_sleep);

    void
    set_model_matrices(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) const;
    std::vector<image<uint8_t>> get_enemy_visions() const;

    void loop_wait() const;
    void kill_threads();

private:
    int num_threads_;

    bool thread_sleep_;
    std::atomic<bool> threads_running_;
    std::vector<std::thread> pool_;

    std::vector<std::shared_ptr<EnemyTankFactory>> enemy_tank_factories_;

    std::unique_ptr<ModelMatricesDoubleBuffer> model_matrices_;
    std::vector<std::unique_ptr<VisionDoubleBuffer>> enemy_visions_;

    std::unique_ptr<std::barrier<>> reset_barrier_;
    std::unique_ptr<std::barrier<>> loop_barrier_;
};

#endif//ARENAI_TRAIN_HOST_THREAD_POOL_H
