//
// Created by samuel on 29/09/2025.
//

#ifndef ARENAI_ENVIRONMENT_H
#define ARENAI_ENVIRONMENT_H

#include <barrier>
#include <future>
#include <memory>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include <arenai_model/engine.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/pbuffer_renderer.h>

#include "./enemy_handler.h"
#include "./enemy_tank_factory.h"
#include "./types.h"

struct VisionDoubleBuffer {
    std::shared_ptr<image<uint8_t>> buf[2];
    std::atomic<int> front{0};

    void init(std::mt19937 &rng, int height, int width);
};

struct ModelMatricesDoubleBuffer {
    std::vector<std::tuple<std::string, glm::mat4>> buf[2];

    std::atomic<int> front_idx{0};
    std::atomic<uint64_t> frame_id{0};
};

class BaseTanksEnvironment {
public:
    BaseTanksEnvironment(
        const std::shared_ptr<AbstractFileReader> &file_reader,
        const std::shared_ptr<AbstractGLContext> &gl_context, int nb_tanks, float wanted_frequency,
        bool thread_sleep);

    virtual std::vector<std::tuple<State, Reward, IsDone>>
    step(float time_delta, std::future<std::vector<Action>> &actions_future);

    std::vector<State> reset_physics();
    void reset_drawables(const std::shared_ptr<AbstractGLContext> &new_gl_context);
    void stop_drawing();

    virtual ~BaseTanksEnvironment();

private:
    float wanted_frequency;
    int nb_tanks;

    bool thread_sleep;
    std::atomic<bool> threads_running;
    std::vector<std::thread> pool;

    ModelMatricesDoubleBuffer model_matrices;

    std::vector<std::unique_ptr<EnemyTankFactory>> tank_factories;
    std::vector<std::unique_ptr<EnemyControllerHandler>> tank_controller_handler;
    std::vector<VisionDoubleBuffer> enemy_visions;

    std::unique_ptr<PhysicEngine> physic_engine;

    std::shared_ptr<AbstractGLContext> gl_context;

    int nb_reset_frames;

    void worker_enemy_vision(int index, const std::unique_ptr<EnemyTankFactory> &tank_factory);

protected:
    virtual void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) = 0;

    virtual void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) = 0;
    virtual void on_reset_drawables(
        const std::unique_ptr<PhysicEngine> &engine,
        const std::shared_ptr<AbstractGLContext> &gl_context) = 0;

    void start_threads();
    void kill_threads();

    template<typename T>
    T apply_on_factories(
        std::function<T(const std::vector<std::unique_ptr<EnemyTankFactory>> &)> apply_function) {
        return apply_function(tank_factories);
    }

    std::vector<std::tuple<std::string, glm::mat4>> publish_and_get_model_matrices();

    std::random_device dev;
    std::mt19937 rng;

    std::shared_ptr<AbstractFileReader> file_reader;
};

#endif// ARENAI_ENVIRONMENT_H
