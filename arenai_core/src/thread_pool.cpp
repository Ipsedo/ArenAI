#include <algorithm>
#include <chrono>
#include <set>

#include <arenai_core/environment.h>
#include <arenai_core/thread_pool.h>
#include <arenai_core/types.h>

using namespace arenai;
using namespace arenai::core;

namespace arenai::core {

    view::image<uint8_t> VisionDoubleBuffer::black_image(const int height, const int width) {
        return {std::vector<uint8_t>(3 * height * width, 0)};
    }

    VisionDoubleBuffer::VisionDoubleBuffer(const int height, const int width)
        : DoubleBuffer(black_image(height, width)) {}

    PanoramaDoubleBuffer::PanoramaDoubleBuffer()
        : DoubleBuffer(
            view::image<uint8_t>{std::vector<uint8_t>(3 * PANORAMA_HEIGHT * PANORAMA_WIDTH, 0)}) {}

    ModelMatricesDoubleBuffer::ModelMatricesDoubleBuffer()
        : DoubleBuffer(std::vector<std::tuple<std::string, glm::mat4>>()) {}

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

    EnemyVisionThreadPool::EnemyVisionThreadPool(
        const int num_tanks, const int max_concurrent_renders, const float wanted_frequency,
        const bool thread_sleep)
        : num_tanks_(num_tanks), wanted_frequency_(wanted_frequency), thread_sleep_(thread_sleep),
          threads_running_(true), limiter_(max_concurrent_renders),
          model_matrices_(std::make_unique<ModelMatricesDoubleBuffer>()),
          reset_barrier_(std::nullptr_t()), start_barrier_(std::nullptr_t()),
          loop_barrier_(std::nullptr_t()) {}

    void EnemyVisionThreadPool::set_seed(const unsigned int seed) { seed_ = seed; }

    void EnemyVisionThreadPool::start_thread(
        const std::vector<std::shared_ptr<model::EnemyTank>> &tank_factories,
        const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::vector<std::tuple<std::string, glm::mat4>> &initial_model_matrices,
        const std::vector<std::shared_ptr<model::Item>> &scene_items) {
        num_tanks_ = static_cast<int>(tank_factories.size());

        threads_running_.store(true, std::memory_order_release);

        reset_barrier_ = std::make_unique<std::barrier<>>(num_tanks_ + 1);
        start_barrier_ = std::make_unique<std::barrier<>>(num_tanks_ + 1);
        loop_barrier_ = std::make_unique<std::barrier<>>(num_tanks_ + 1);

        model_matrices_->write(initial_model_matrices);

        enemy_visions_.clear();
        enemy_visions_.reserve(num_tanks_);
        for (int i = 0; i < num_tanks_; i++)
            enemy_visions_.push_back(
                std::make_unique<VisionDoubleBuffer>(VISION_HEIGHT, VISION_WIDTH));

        enemy_panoramas_.clear();
        enemy_panoramas_.reserve(num_tanks_);
        for (int i = 0; i < num_tanks_; i++)
            enemy_panoramas_.push_back(std::make_unique<PanoramaDoubleBuffer>());

        const auto drawable_factory = graphics_backend->drawable_factory();

        SharedScene scene{.sky = drawable_factory->make_cube_texture(file_reader, "cubemap/1")};
        for (const auto &item: scene_items)
            scene.item_meshes.emplace_back(
                item->get_name(), drawable_factory->make_mesh(item->get_shape()->get_vertices()));
        for (const auto &tank_factory: tank_factories)
            for (const auto &item: tank_factory->get_items())
                scene.tank_parts.insert(item->get_name());
        for (const auto &tank_factory: tank_factories)
            for (const auto &[name, shape]: tank_factory->load_shell_shapes())
                if (!scene.shell_meshes.contains(name))
                    scene.shell_meshes.emplace(
                        name, drawable_factory->make_mesh(shape->get_vertices()));

        pool_.reserve(num_tanks_);
        for (int i = 0; i < num_tanks_; i++)
            pool_.emplace_back([this, i, &tank_factories, graphics_backend, scene] {
                worker_loop(tank_factories[i], graphics_backend, scene, i);
            });
    }

    void EnemyVisionThreadPool::worker_loop(
        const std::shared_ptr<model::EnemyTank> &tank_factory,
        const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend,
        const SharedScene &scene, const int index) {

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

        auto renderer = graphics_backend->make_offscreen_renderer(
            VISION_WIDTH, VISION_HEIGHT, glm::vec3(200, 300, 200), tank_factory->get_camera());

        renderer->make_current();

        const auto drawable_factory = graphics_backend->drawable_factory();

        std::uniform_real_distribution u_dist(0.f, 1.f);

        renderer->add_drawable("cubemap", drawable_factory->make_cube_map(scene.sky));

        // every view of this tank draws an item with the same color
        std::vector<std::tuple<std::string, std::shared_ptr<view::AbstractMesh>, glm::vec4>>
            colored;

        for (const auto &[name, mesh]: scene.item_meshes) {
            const glm::vec4 color(u_dist(local_rng), u_dist(local_rng), u_dist(local_rng), 1.f);
            colored.emplace_back(name, mesh, color);
        }

        for (const auto &[name, mesh]: scene.shell_meshes) {
            const glm::vec4 shell_color(
                u_dist(local_rng), u_dist(local_rng), u_dist(local_rng), 1.f);
            colored.emplace_back(name, mesh, shell_color);
        }

        for (const auto &[name, mesh, color]: colored) {
            renderer->add_drawable(name, drawable_factory->make_diffuse(mesh, color));
            if (scene.tank_parts.contains(name))
                renderer->add_drawable(
                    wreck_drawable_name(name), drawable_factory->make_hatched(mesh, color));
        }

        // the panorama leaves out the tank's own parts
        std::set<std::string> own_parts;
        for (const auto &item: tank_factory->get_items()) {
            own_parts.insert(item->get_name());
            own_parts.insert(wreck_drawable_name(item->get_name()));
        }

        constexpr int view_width = PANORAMA_WIDTH / PANORAMA_VIEWS;

        std::vector<std::unique_ptr<view::AbstractOffscreenRenderer>> panorama_renderers;
        for (const auto &camera: tank_factory->get_panorama_cameras()) {
            auto panorama_renderer = graphics_backend->make_offscreen_renderer(
                view_width, PANORAMA_HEIGHT, glm::vec3(200, 300, 200), camera);

            panorama_renderer->make_current();

            panorama_renderer->add_drawable("cubemap", drawable_factory->make_cube_map(scene.sky));

            for (const auto &[name, mesh, color]: colored) {
                if (own_parts.contains(name)) continue;
                panorama_renderer->add_drawable(name, drawable_factory->make_diffuse(mesh, color));
                if (scene.tank_parts.contains(name))
                    panorama_renderer->add_drawable(
                        wreck_drawable_name(name), drawable_factory->make_hatched(mesh, color));
            }

            panorama_renderers.push_back(std::move(panorama_renderer));
        }

        const auto frame_dt =
            std::chrono::milliseconds(static_cast<int>(wanted_frequency_ * 1000.f));

        while (threads_running_.load(std::memory_order_acquire)) {
            start_barrier_->arrive_and_wait();

            if (!threads_running_.load(std::memory_order_acquire)) break;

            auto last_time = std::chrono::steady_clock::now();

            limiter_.acquire();

            const auto matrices = model_matrices_->read_copy();

            renderer->make_current();
            enemy_visions_[index]->write(renderer->draw_and_get_frame(matrices));

            std::vector<std::tuple<std::string, glm::mat4>> seen_matrices;
            seen_matrices.reserve(matrices.size());
            for (const auto &matrix: matrices)
                if (!own_parts.contains(std::get<0>(matrix))) seen_matrices.push_back(matrix);

            // [C, H, W] strip: the views side by side
            view::image<uint8_t> panorama{
                std::vector<uint8_t>(3 * PANORAMA_HEIGHT * PANORAMA_WIDTH)};
            for (int v = 0; v < static_cast<int>(panorama_renderers.size()); v++) {
                panorama_renderers[v]->make_current();
                const auto frame = panorama_renderers[v]->draw_and_get_frame(seen_matrices);

                for (int row = 0; row < 3 * PANORAMA_HEIGHT; row++)
                    std::copy_n(
                        frame.pixels.begin() + row * view_width, view_width,
                        panorama.pixels.begin() + row * PANORAMA_WIDTH + v * view_width);
            }
            enemy_panoramas_[index]->write(std::move(panorama));

            limiter_.release();

            auto now = std::chrono::steady_clock::now();
            auto dt = now - last_time;

            loop_barrier_->arrive_and_wait();

            if (thread_sleep_)
                std::this_thread::sleep_for(
                    std::max(frame_dt - dt, std::chrono::steady_clock::duration::zero()));
        }

        panorama_renderers.clear();
        renderer.reset();
        graphics_backend->release_thread();

        start_barrier_->arrive_and_drop();
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

    view::image<uint8_t> EnemyVisionThreadPool::read_panorama(const int index) const {
        return enemy_panoramas_[index]->read_copy();
    }

    void EnemyVisionThreadPool::begin_frame() const { (void) start_barrier_->arrive(); }

    void EnemyVisionThreadPool::loop_wait() const { loop_barrier_->arrive_and_wait(); }

    void EnemyVisionThreadPool::kill_threads() {
        if (pool_.empty()) return;

        threads_running_.store(false, std::memory_order_release);

        start_barrier_->arrive_and_drop();
        loop_barrier_->arrive_and_drop();
        reset_barrier_->arrive_and_wait();

        for (auto &t: pool_)
            if (t.joinable()) t.join();

        pool_.clear();
    }

    EnemyVisionThreadPool::~EnemyVisionThreadPool() { kill_threads(); }

}// namespace arenai::core
