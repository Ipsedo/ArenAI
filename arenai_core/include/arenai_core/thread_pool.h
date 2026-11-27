#ifndef ARENAI_AGENT_HOST_THREAD_POOL_H
#define ARENAI_AGENT_HOST_THREAD_POOL_H
#include <barrier>
#include <condition_variable>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <thread>
#include <vector>

#include <arenai_model/engine.h>
#include <arenai_model/tank.h>
#include <arenai_utils/double_buffer.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/backend.h>
#include <arenai_view/renderer.h>

namespace arenai::core {
    class VisionDoubleBuffer : public utils::DoubleBuffer<view::image<uint8_t>> {
    public:
        VisionDoubleBuffer(int height, int width);

    private:
        static view::image<uint8_t> black_image(int height, int width);
    };

    class PanoramaDoubleBuffer : public utils::DoubleBuffer<view::image<uint8_t>> {
    public:
        PanoramaDoubleBuffer();
    };

    class ModelMatricesDoubleBuffer
        : public utils::DoubleBuffer<std::vector<std::tuple<std::string, glm::mat4>>> {
    public:
        ModelMatricesDoubleBuffer();
    };

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
            int num_tanks, int max_concurrent_renders, float wanted_frequency, bool thread_sleep);

        void set_model_matrices(
            const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) const;

        view::image<uint8_t> read_vision(int index) const;
        view::image<uint8_t> read_panorama(int index) const;

        void set_seed(unsigned int seed);

        void start_thread(
            const std::vector<std::shared_ptr<model::EnemyTank>> &tank_factories,
            const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::vector<std::tuple<std::string, glm::mat4>> &initial_model_matrices,
            const std::vector<std::shared_ptr<model::Item>> &scene_items);

        void begin_frame() const;
        void loop_wait() const;
        void kill_threads();

        ~EnemyVisionThreadPool();

    private:
        int num_tanks_;

        float wanted_frequency_;
        bool thread_sleep_;
        std::atomic<bool> threads_running_;
        std::vector<std::thread> pool_;

        ThreadLimiter limiter_;

        std::unique_ptr<ModelMatricesDoubleBuffer> model_matrices_;
        std::vector<std::unique_ptr<VisionDoubleBuffer>> enemy_visions_;
        std::vector<std::unique_ptr<PanoramaDoubleBuffer>> enemy_panoramas_;

        std::unique_ptr<std::barrier<>> reset_barrier_;
        std::unique_ptr<std::barrier<>> start_barrier_;
        std::unique_ptr<std::barrier<>> loop_barrier_;

        std::optional<unsigned int> seed_;

        std::random_device dev_;

        // GPU resources uploaded once, drawn by every tank's renderers
        struct SharedScene {
            std::shared_ptr<view::AbstractCubeTexture> sky;
            std::vector<std::tuple<std::string, std::shared_ptr<view::AbstractMesh>>> item_meshes;
            std::map<std::string, std::shared_ptr<view::AbstractMesh>> shell_meshes;
        };

        void worker_loop(
            const std::shared_ptr<model::EnemyTank> &tank_factory,
            const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend,
            const SharedScene &scene, int index);
    };
}// namespace arenai::core

#endif//ARENAI_AGENT_HOST_THREAD_POOL_H
