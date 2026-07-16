//
// Created by samuel on 11/03/2026.
//

#include "./game_environment.h"

#include <iostream>

#include <arenai_model/engine.h>
#include <arenai_model/tank.h>
#include <arenai_model/tank_factory.h>
#include <arenai_train/file_reader.h>

using namespace arenai;

namespace arenai::desktop {

    DesktopGameEnvironment::DesktopGameEnvironment(
        const std::filesystem::path &asset_folder_path,
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
        const int nb_tanks, const int vision_height, const int vision_width,
        const float wanted_frequency)
        // The tank visions get their own headless backend (integrated GPU): their
        // synchronous readbacks are latency-bound on a discrete GPU, and this keeps
        // them off the window's GPU when the player view is offloaded (prime-run).
        : core::BaseTanksEnvironment(
            std::make_shared<train::DesktopAssetFileReader>(asset_folder_path),
            view::make_opengl_backend(), nb_tanks, wanted_frequency, vision_height, vision_width, 8,
            true),
          windowed_backend(graphics_backend),
          asset_file_reader(std::make_shared<train::DesktopAssetFileReader>(asset_folder_path)),
          player_tank(std::nullptr_t()), player_renderer(std::nullptr_t()),
          player_controller_handler(std::nullptr_t()), wanted_frequency(wanted_frequency) {}

    void DesktopGameEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
        // the base environment leaves its own (headless) context current on this
        // thread, so bind the window's context before drawing the player view
        player_renderer->make_current();
        player_renderer->draw(model_matrices);
    }

    void DesktopGameEnvironment::on_reset_physics(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {
        player_tank = engine->get_tank_factory()->make_player_tank(
            file_reader, "player", glm::vec3(0., -40., 40));
    }

    void DesktopGameEnvironment::on_reset_drawables(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {
        player_renderer = windowed_backend->make_player_renderer(
            glm::vec3(200, 300, 200), player_tank->get_camera());

        /*player_controller_handler = std::make_shared<PlayerMouseKeyboardHandler>(
            windowed_backend->get_window(), *player_renderer);*/
        player_controller_handler = std::make_shared<PlayerGamepadHandler>();

        for (auto &ctrl: player_tank->get_controllers())
            player_controller_handler->add_controller(ctrl);

        windowed_backend->get_window()->set_gamepad_callback(player_controller_handler);

        windowed_backend->get_window()->set_resize_callback(
            [this](const int width, const int height) -> void {
                player_renderer->set_window_size(width, height);
            });

        player_renderer->make_current();

        const auto drawable_factory = windowed_backend->drawable_factory();

        player_renderer->add_drawable(
            "cubemap", drawable_factory->make_cube_map(file_reader, "cubemap/1"));

        std::uniform_real_distribution<float> u_dist(0.f, 1.f);

        for (const auto &[name, shape]: player_tank->load_shell_shapes()) {
            const glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

            player_renderer->add_drawable(
                name, drawable_factory->make_diffuse(file_reader, shape->get_vertices(), color));
        }

        for (const auto &item: engine->get_items()) {
            const glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

            player_renderer->add_drawable(
                item->get_name(), drawable_factory->make_diffuse(
                                      file_reader, item->get_shape()->get_vertices(), color));
        }

        /*for (auto &hud_drawable: player_controller_handler->get_hud_drawables(file_reader))
        player_renderer->add_hud_drawable(std::move(hud_drawable));*/

        player_renderer->release_current();
    }

    DesktopGameEnvironment::~DesktopGameEnvironment() {
        std::cout << "Final score : " << player_tank->get_score() << std::endl;
    }

}// namespace arenai::desktop
