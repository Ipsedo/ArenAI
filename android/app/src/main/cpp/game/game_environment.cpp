//
// Created by samuel on 21/03/2023.
//

#include "./game_environment.h"

#include <glm/gtx/transform.hpp>

#include <arenai_model/engine.h>
#include <arenai_model/shapes.h>
#include <arenai_model/tank.h>
#include <arenai_model/tank_factory.h>
#include <arenai_utils/cache.h>
#include <arenai_utils/logging.h>
#include <arenai_utils/singleton.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/diffuse.h>

#include "../utils/android_file_reader.h"
#include "./android_gl_context.h"

UserGameTanksEnvironment::UserGameTanksEnvironment(
    struct android_app *app, EGLDisplay display, int nb_tanks, float wanted_frequency)
    : BaseTanksEnvironment(
        std::make_shared<AndroidFileReader>(app->activity->assetManager),
        std::make_shared<AndroidGLContext>(app->window, eglGetDisplay(EGL_DEFAULT_DISPLAY)),
        nb_tanks, wanted_frequency, true),
      app(app), display(eglGetDisplay(EGL_DEFAULT_DISPLAY)), tank_factory(std::nullptr_t()),
      is_paused(true), player_renderer(std::nullptr_t()),
      player_controller_handler(std::nullptr_t()), wanted_frequency(wanted_frequency) {}

void UserGameTanksEnvironment::on_draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
    player_renderer->draw(model_matrices);
}

int32_t UserGameTanksEnvironment::on_input(struct android_app *app, AInputEvent *event) {
    if (AKeyEvent_getKeyCode(event) == AKEYCODE_BACK) {
        ANativeActivity_finish(app->activity);
        return 1;
    }

    return player_controller_handler->on_event(event);
}

void UserGameTanksEnvironment::on_cmd(struct android_app *app, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_SAVE_STATE:
            // The system has asked us to save our current state.  Do so.
            /*app->savedState = malloc(sizeof(UserGameTanksEnvironment));
            // TODO real object copy...
            app->savedState = engine;
            app->savedStateSize = sizeof(UserGameTanksEnvironment);
            */
            LOG_INFO("save state");
            break;
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            if (app->window != nullptr) {
                LOG_INFO("opening window");
                // TODO fix resuming
                reset_drawables(std::make_shared<AndroidGLContext>(
                    app->window, eglGetDisplay(EGL_DEFAULT_DISPLAY)));
                is_paused = false;
            }
            break;
        case APP_CMD_PAUSE:
            pause();
            LOG_INFO("close");
            break;
        case APP_CMD_GAINED_FOCUS:
            is_paused = false;
            LOG_INFO("gained focus");
            break;
        default: break;
    }
}

bool UserGameTanksEnvironment::is_running() const { return !is_paused; }

void UserGameTanksEnvironment::on_reset_physics(
    const std::unique_ptr<AbstractPhysicEngine> &engine) {
    auto tf = make_tank_factory(*engine, file_reader, wanted_frequency);
    tank_factory = tf->make_player_tank("player", glm::vec3(0., -40., 40));
}

void UserGameTanksEnvironment::on_reset_drawables(
    const std::unique_ptr<AbstractPhysicEngine> &engine,
    const std::shared_ptr<AbstractGLContext> &gl_context) {
    player_renderer = std::make_unique<PlayerRenderer>(
        gl_context, ANativeWindow_getWidth(app->window), ANativeWindow_getHeight(app->window),
        glm::vec3(200, 300, 200), tank_factory->get_camera());
    player_renderer->make_current();

    player_controller_handler = std::make_unique<PlayerControllerHandler>(
        app->config, player_renderer->get_width(), player_renderer->get_height());

    for (auto &ctrl: tank_factory->get_controllers())
        player_controller_handler->add_controller(ctrl);

    player_renderer->add_drawable("cubemap", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    std::uniform_real_distribution<float> u_dist(0.f, 1.f);

    for (const auto &[name, shape]: tank_factory->load_shell_shapes()) {
        glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

        player_renderer->add_drawable(
            name, std::make_unique<Diffuse>(file_reader, shape->get_vertices(), color));
    }

    for (const auto &item: engine->get_items()) {
        glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

        player_renderer->add_drawable(
            item->get_name(),
            std::make_unique<Diffuse>(file_reader, item->get_shape()->get_vertices(), color));
    }

    for (auto &hud_drawable: player_controller_handler->get_hud_drawables(file_reader))
        player_renderer->add_hud_drawable(std::move(hud_drawable));

    player_renderer->release_current();
}

void UserGameTanksEnvironment::pause() {
    is_paused = true;
    stop_drawing();
}

void UserGameTanksEnvironment::reset_singleton() {
    Singleton<Cache<std::shared_ptr<Shape>>>::get_singleton()->clear();
    Singleton<Cache<std::shared_ptr<Shape>>>::reset_singleton();

    Singleton<Cache<std::shared_ptr<Program>>>::get_singleton()->clear();
    Singleton<Cache<std::shared_ptr<Program>>>::reset_singleton();
}
