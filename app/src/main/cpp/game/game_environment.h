//
// Created by samuel on 21/03/2023.
//

#ifndef ARENAI_GAME_ENVIRONMENT_H
#define ARENAI_GAME_ENVIRONMENT_H

#include <random>

#include <android_native_app_glue.h>

#include <arenai_controller/controller.h>
#include <arenai_core/environment.h>
#include <arenai_model/engine.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/renderer.h>

#include "../controller/player_handler.h"
#include "./android_gl_context.h"
#include "./player_tank_factory.h"

class UserGameTanksEnvironment : public BaseTanksEnvironment {
public:
    explicit UserGameTanksEnvironment(
        struct android_app *app, EGLDisplay display, int nb_tanks, float wanted_frequency);

    bool is_running() const;

    int32_t on_input(struct android_app *new_app, AInputEvent *event);

    void on_cmd(struct android_app *app, int32_t cmd);

    void pause();

    static void reset_singleton();

protected:
    void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

    void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) override;

    void on_reset_drawables(
        const std::unique_ptr<PhysicEngine> &engine,
        const std::shared_ptr<AbstractGLContext> &gl_context) override;

private:
    android_app *app;
    EGLDisplay display;

    std::unique_ptr<TankFactory> tank_factory;

    std::unique_ptr<PlayerRenderer> player_renderer;
    std::unique_ptr<PlayerControllerHandler> player_controller_handler;

    bool is_paused;

    float wanted_frequency;
};

#endif// ARENAI_GAME_ENVIRONMENT_H
