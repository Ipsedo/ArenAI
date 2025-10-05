//
// Created by samuel on 29/09/2025.
//

#ifndef PHYVR_ENVIRONMENT_H
#define PHYVR_ENVIRONMENT_H

#include <barrier>
#include <memory>
#include <random>
#include <thread>
#include <tuple>

#include <phyvr_controller/inputs.h>
#include <phyvr_model/engine.h>
#include <phyvr_model/tank_factory.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/framebuffer_renderer.h>

struct State {
    std::vector<std::vector<pixel>> vision;
    std::vector<float> proprioception;
};

typedef float Reward;

typedef bool IsFinish;

typedef user_input Action;

class BaseTanksEnvironment {
public:
    BaseTanksEnvironment(
        const std::shared_ptr<AbstractFileReader> &file_reader,
        const std::shared_ptr<AbstractGLContext> &gl_context, int nb_tanks);

    virtual std::vector<std::tuple<State, Reward, IsFinish>>
    step(float time_delta, const std::vector<Action> &actions);

    std::vector<State> reset_physics();
    void reset_drawables(const std::shared_ptr<AbstractGLContext> &new_gl_context);

    virtual ~BaseTanksEnvironment();

private:
    int nb_tanks;

    std::atomic<bool> threads_running;
    std::unique_ptr<std::barrier<>> thread_barrier;
    std::vector<std::thread> pool;

    std::vector<std::tuple<std::string, glm::mat4>> model_matrices;

    std::vector<std::unique_ptr<TankFactory>> tank_factories;
    std::vector<std::unique_ptr<PBufferRenderer>> tank_renderers;
    std::vector<std::vector<std::vector<pixel>>> enemy_visions;

    std::unique_ptr<PhysicEngine> physic_engine;

    std::shared_ptr<AbstractGLContext> gl_context;

    void worker_enemy_vision(int index, std::unique_ptr<PBufferRenderer> &renderer);

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
