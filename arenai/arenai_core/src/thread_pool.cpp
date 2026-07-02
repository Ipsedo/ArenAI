//
// Created by samuel on 10/06/2026.
//

#include <chrono>

#include <arenai_core/constants.h>
#include <arenai_core/thread_pool.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/specular.h>

using namespace arenai;
using namespace arenai::core;

namespace arenai::core {

    /*
 * VisionDoubleBuffer / ModelMatricesDoubleBuffer
 */

    view::image<uint8_t> VisionDoubleBuffer::black_image(const int height, const int width) {
        return {std::vector<uint8_t>(3 * height * width, 0)};
    }

    VisionDoubleBuffer::VisionDoubleBuffer(const int height, const int width)
        : DoubleBuffer(black_image(height, width)) {}

    ModelMatricesDoubleBuffer::ModelMatricesDoubleBuffer()
        : DoubleBuffer(std::vector<std::tuple<std::string, glm::mat4>>()) {}

    /*
 * ThreadLimiter
 */

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
 * EnemyVisionThreadPool
 */

    EnemyVisionThreadPool::EnemyVisionThreadPool(
        const int num_tanks, const int max_concurrent_renders, const int vision_height,
        const int vision_width, const float wanted_frequency, const bool thread_sleep)
        : num_tanks_(num_tanks), vision_height_(vision_height), vision_width_(vision_width),
          wanted_frequency_(wanted_frequency), thread_sleep_(thread_sleep), threads_running_(true),
          limiter_(max_concurrent_renders),
          model_matrices_(std::make_unique<ModelMatricesDoubleBuffer>()),
          reset_barrier_(std::nullptr_t()), loop_barrier_(std::nullptr_t()) {}

    void EnemyVisionThreadPool::set_seed(const unsigned int seed) { seed_ = seed; }

    void EnemyVisionThreadPool::start_thread(
        const std::vector<std::shared_ptr<model::EnemyTank>> &tank_factories,
        const std::shared_ptr<view::AbstractGLContext> &gl_context,
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::vector<std::tuple<std::string, glm::mat4>> &initial_model_matrices,
        const std::vector<std::shared_ptr<model::Item>> &scene_items) {
        num_tanks_ = static_cast<int>(tank_factories.size());

        threads_running_.store(true, std::memory_order_release);

        reset_barrier_ = std::make_unique<std::barrier<>>(num_tanks_ + 1);
        loop_barrier_ = std::make_unique<std::barrier<>>(num_tanks_ + 1);

        model_matrices_->write(initial_model_matrices);

        enemy_visions_.clear();
        enemy_visions_.reserve(num_tanks_);
        for (int i = 0; i < num_tanks_; i++)
            enemy_visions_.push_back(
                std::make_unique<VisionDoubleBuffer>(vision_height_, vision_width_));

        pool_.reserve(num_tanks_);
        for (int i = 0; i < num_tanks_; i++)
            pool_.emplace_back([this, i, &tank_factories, gl_context, file_reader, scene_items] {
                worker_loop(tank_factories[i], gl_context, file_reader, scene_items, i);
            });
    }

    void EnemyVisionThreadPool::worker_loop(
        const std::shared_ptr<model::EnemyTank> &tank_factory,
        const std::shared_ptr<view::AbstractGLContext> &gl_context,
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::vector<std::shared_ptr<model::Item>> &scene_items, const int index) {

        std::mt19937 local_rng;
        if (seed_.has_value()) {
            std::seed_seq seq{seed_.value(), static_cast<uint32_t>(index)};
            local_rng.seed(seq);
        } else {
            std::seed_seq seq{
                dev_(), static_cast<uint32_t>(reinterpret_cast<uintptr_t>(this)),
                static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tank_factory.get())),
                static_cast<uint32_t>(index),
                static_cast<uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count())};
            local_rng.seed(seq);
        }

        auto renderer = std::make_unique<view::PBufferRenderer>(
            gl_context, vision_width_, vision_height_, glm::vec3(200, 300, 200),
            tank_factory->get_camera());

        renderer->make_current();

        std::uniform_real_distribution u_dist(0.f, 1.f);

        renderer->add_drawable(
            "cubemap", std::make_unique<view::CubeMap>(file_reader, "cubemap/1"));

        for (const auto &item: scene_items) {
            glm::vec4 color(u_dist(local_rng), u_dist(local_rng), u_dist(local_rng), 1.f);
            const auto shape = item->get_shape();
            renderer->add_drawable(
                item->get_name(), std::make_unique<view::Specular>(
                                      file_reader, shape->get_vertices(), shape->get_normals(),
                                      color, color, color, 50.f));
        }

        for (const auto &[name, shape]: tank_factory->load_shell_shapes()) {
            glm::vec4 shell_color(u_dist(local_rng), u_dist(local_rng), u_dist(local_rng), 1.f);

            renderer->add_drawable(
                name, std::make_unique<view::Specular>(
                          file_reader, shape->get_vertices(), shape->get_normals(), shell_color,
                          shell_color, shell_color, 50.f));
        }
        const auto frame_dt =
            std::chrono::milliseconds(static_cast<int>(wanted_frequency_ * 1000.f));

        while (threads_running_.load(std::memory_order_acquire)) {
            auto last_time = std::chrono::steady_clock::now();

            limiter_.acquire();

            const auto matrices = model_matrices_->read_copy();
            enemy_visions_[index]->write(renderer->draw_and_get_frame(matrices));

            limiter_.release();

            auto now = std::chrono::steady_clock::now();
            auto dt = now - last_time;

            loop_barrier_->arrive_and_wait();

            if (thread_sleep_)
                std::this_thread::sleep_for(
                    std::max(frame_dt - dt, std::chrono::steady_clock::duration::zero()));
        }

        renderer.reset();
        eglReleaseThread();

        loop_barrier_->arrive_and_drop();
        reset_barrier_->arrive_and_wait();
    }

    void EnemyVisionThreadPool::set_model_matrices(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) const {
        model_matrices_->write(model_matrices);
    }

    view::image<uint8_t> EnemyVisionThreadPool::read_vision(const int index) const {
        return enemy_visions_[index]->read_copy();
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

    EnemyVisionThreadPool::~EnemyVisionThreadPool() { kill_threads(); }

}// namespace arenai::core
