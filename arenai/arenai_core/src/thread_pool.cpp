//
// Created by samuel on 10/06/2026.
//

#include <arenai_core/constants.h>
#include <arenai_core/thread_pool.h>

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

/*
 * Pool
 */

EnemyVisionThreadPool::EnemyVisionThreadPool(
    const int num_threads,
    const std::vector<std::shared_ptr<EnemyTankFactory>> &enemy_tank_factories,
    const bool thread_sleep)
    : num_threads_(num_threads), thread_sleep_(thread_sleep), threads_running_(false),
      enemy_tank_factories_(enemy_tank_factories),
      model_matrices_(std::make_unique<ModelMatricesDoubleBuffer>()),
      reset_barrier_(std::make_unique<std::barrier<>>(num_threads_ + 1)),
      loop_barrier_(std::make_unique<std::barrier<>>(num_threads_ + 1)) {

    enemy_visions_.clear();
    enemy_visions_.reserve(enemy_tank_factories_.size());
    for (int i = 0; i < enemy_tank_factories_.size(); i++)
        enemy_visions_.push_back(
            std::make_unique<VisionDoubleBuffer>(ENEMY_VISION_HEIGHT, ENEMY_VISION_WIDTH));

    threads_running_.store(true, std::memory_order_release);
    pool_.clear();
    pool_.reserve(enemy_tank_factories_.size());

    for (int i = 0; i < enemy_tank_factories_.size(); ++i) pool_.emplace_back([this, i] {});
}

void EnemyVisionThreadPool::set_model_matrices(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) const {
    model_matrices_->write(model_matrices);
}

std::vector<image<uint8_t>> EnemyVisionThreadPool::get_enemy_visions() const {
    std::vector<image<uint8_t>> result;
    result.reserve(enemy_tank_factories_.size());

    for (int i = 0; i < enemy_tank_factories_.size(); i++)
        result.push_back(enemy_visions_[i]->read_copy());

    return result;
}

void EnemyVisionThreadPool::loop_wait() const { loop_barrier_->arrive_and_wait(); }

void EnemyVisionThreadPool::kill_threads() {
    if (pool_.empty()) return;

    threads_running_.store(false, std::memory_order_release);

    loop_barrier_->arrive_and_drop();
    reset_barrier_->arrive_and_wait();

    for (auto &t: pool_)
        if (t.joinable()) t.join();

    pool_.clear();
}
