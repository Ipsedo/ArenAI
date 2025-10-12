//
// Created by samuel on 29/09/2025.
//

#ifndef PHYVR_ENVIRONMENT_H
#define PHYVR_ENVIRONMENT_H

#include <barrier>
#include <future>
#include <memory>
#include <random>
#include <shared_mutex>
#include <thread>
#include <tuple>

#include <phyvr_model/engine.h>
#include <phyvr_model/tank_factory.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/pbuffer_renderer.h>

#include "./enemy_handler.h"
#include "./types.h"

class BaseTanksEnvironment {
public:
    BaseTanksEnvironment(
        const std::shared_ptr<AbstractFileReader> &file_reader,
        const std::shared_ptr<AbstractGLContext> &gl_context, int nb_tanks, float wanted_frequency);

    virtual std::vector<std::tuple<State, Reward, IsFinish>>
    step(float time_delta, std::future<std::vector<Action>> &actions_future);

    std::vector<State> reset_physics();
    void reset_drawables(const std::shared_ptr<AbstractGLContext> &new_gl_context);

    virtual ~BaseTanksEnvironment();

private:
    float wanted_frequency;
    int nb_tanks;
    std::vector<std::mutex> visions_mutex;

    std::atomic<bool> threads_running;
    std::unique_ptr<std::barrier<>> thread_barrier;
    std::vector<std::thread> pool;

    std::shared_mutex model_matrices_mutex;
    std::vector<std::tuple<std::string, glm::mat4>> model_matrices;

    std::vector<std::unique_ptr<EnemyTankFactory>> tank_factories;
    std::vector<std::unique_ptr<PBufferRenderer>> tank_renderers;
    std::vector<std::unique_ptr<EnemyControllerHandler>> tank_controller_handler;
    std::vector<image<uint8_t>> enemy_visions;

    std::unique_ptr<PhysicEngine> physic_engine;

    std::shared_ptr<AbstractGLContext> gl_context;

    void worker_enemy_vision(int index, const std::unique_ptr<PBufferRenderer> &renderer);

protected:
    virtual void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) = 0;

    virtual void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) = 0;
    virtual void on_reset_drawables(
        const std::unique_ptr<PhysicEngine> &engine,
        const std::shared_ptr<AbstractGLContext> &gl_context) = 0;

    void start_threads();
    void kill_threads();

    std::random_device dev;
    std::mt19937 rng;

    std::shared_ptr<AbstractFileReader> file_reader;
};

#endif// PHYVR_ENVIRONMENT_H
