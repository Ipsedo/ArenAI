//
// Created by samuel on 29/09/2025.
//

#ifndef ARENAI_ENVIRONMENT_H
#define ARENAI_ENVIRONMENT_H

#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <arenai_model/engine.h>
#include <arenai_model/tank.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/backend.h>
#include <arenai_view/renderer.h>

#include "./enemy_handler.h"
#include "./thread_pool.h"
#include "./types.h"

namespace arenai::core {
    class BaseTanksEnvironment {
    public:
        BaseTanksEnvironment(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend, int nb_tanks,
            float wanted_frequency, int vision_height, int vision_width, int vision_num_threads,
            bool vision_thread_sleep);

        virtual std::vector<std::tuple<State, Reward, IsDone, IsTruncated>>
        step(float time_delta, const std::vector<Action> &actions);

        std::vector<State> reset(float spawn_width, float spawn_height);

        void seed(unsigned int seed);
        void stop_drawing() const;

        virtual ~BaseTanksEnvironment();

    private:
        float wanted_frequency;
        int nb_tanks;

        int vision_height;
        int vision_width;

        int vision_num_threads;
        bool vision_thread_sleep;

        std::unique_ptr<EnemyVisionThreadPool> vision_pool_;

        std::vector<std::shared_ptr<model::EnemyTank>> tanks;
        std::vector<std::unique_ptr<EnemyControllerHandler>> tank_controller_handler;

        std::unique_ptr<model::AbstractPhysicEngine> physic_engine;

        int nb_reset_frames;

        bool drawing_started_;

        std::shared_ptr<view::AbstractGraphicBackend> graphics_backend;
        std::shared_ptr<view::AbstractRenderContext> gl_context;

        void reset_physics(float spawn_width, float spawn_height);
        void reset_drawables();

    protected:
        std::random_device dev;
        std::mt19937 rng;
        std::shared_ptr<utils::AbstractResourceFileReader> file_reader;

        virtual void
        on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) = 0;

        virtual void
        on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) = 0;
        virtual void
        on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &engine) = 0;

        const std::shared_ptr<view::AbstractGraphicBackend> &get_graphics_backend() const;

        template<typename T>
        T
        apply_on_factories(std::function<T(const std::vector<std::shared_ptr<model::EnemyTank>> &)>
                               apply_function) {
            return apply_function(tanks);
        }

        std::vector<std::tuple<std::string, glm::mat4>> get_model_matrices() const;
    };
}// namespace arenai::core

#endif// ARENAI_ENVIRONMENT_H
