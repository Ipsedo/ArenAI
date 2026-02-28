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
#include <arenai_utils/locked_buffer.h>
#include <arenai_view/pbuffer_renderer.h>

#include "./enemy_handler.h"
#include "./enemy_tank_factory.h"
#include "./types.h"

class VisionDoubleBuffer : public LockedBuffer<image<uint8_t>> {
public:
    VisionDoubleBuffer(std::mt19937 &rng, int height, int width);

private:
    static image<uint8_t> random_image(std::mt19937 &rng, int height, int width);
};

class ModelMatricesDoubleBuffer
    : public LockedBuffer<std::vector<std::tuple<std::string, glm::mat4>>> {
public:
    ModelMatricesDoubleBuffer();
};

class BaseTanksEnvironment {
public:
    BaseTanksEnvironment(
        const std::shared_ptr<AbstractFileReader> &file_reader,
        const std::shared_ptr<AbstractGLContext> &gl_context, int nb_tanks, float wanted_frequency,
        bool thread_sleep);

    virtual std::vector<std::tuple<State, Reward, IsDone>>
    step(float time_delta, const std::vector<Action> &actions);

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

    std::unique_ptr<ModelMatricesDoubleBuffer> model_matrices;

    std::vector<std::unique_ptr<EnemyTankFactory>> tank_factories;
    std::vector<std::unique_ptr<EnemyControllerHandler>> tank_controller_handler;
    std::vector<std::unique_ptr<VisionDoubleBuffer>> enemy_visions;

    std::unique_ptr<PhysicEngine> physic_engine;

    std::shared_ptr<AbstractGLContext> gl_context;

    int nb_reset_frames;

    std::unique_ptr<std::barrier<>> reset_barrier;

    void worker_enemy_vision(int index, const std::unique_ptr<EnemyTankFactory> &tank_factory);

    void start_threads();
    void kill_threads();

protected:
    std::random_device dev;
    std::mt19937 rng;
    std::shared_ptr<AbstractFileReader> file_reader;

    virtual void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) = 0;

    virtual void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) = 0;
    virtual void on_reset_drawables(
        const std::unique_ptr<PhysicEngine> &engine,
        const std::shared_ptr<AbstractGLContext> &gl_context) = 0;

    template<typename T>
    T apply_on_factories(
        std::function<T(const std::vector<std::unique_ptr<EnemyTankFactory>> &)> apply_function) {
        return apply_function(tank_factories);
    }

    std::vector<std::tuple<std::string, glm::mat4>> publish_and_get_model_matrices();
};

#endif// ARENAI_ENVIRONMENT_H
