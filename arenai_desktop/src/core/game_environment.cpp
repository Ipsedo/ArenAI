//
// Created by samuel on 11/03/2026.
//

#include "./game_environment.h"

#include <algorithm>
#include <iostream>

#include <arenai_agent/file_reader.h>
#include <arenai_model/engine.h>
#include <arenai_model/tank.h>
#include <arenai_model/tank_factory.h>

using namespace arenai;

namespace arenai::desktop {

    DesktopGameEnvironment::DesktopGameEnvironment(
        const std::filesystem::path &asset_folder_path,
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
        const int nb_tanks, const int vision_height, const int vision_width,
        const float wanted_frequency, const ControllerKind &controller_kind,
        const ControlBindings &bindings)
        : BaseTanksEnvironment(
            std::make_shared<agent::DesktopAssetFileReader>(asset_folder_path),
            view::make_vulkan_backend(), nb_tanks, wanted_frequency, vision_height, vision_width, 8,
            true),
          windowed_backend(graphics_backend),
          asset_file_reader(std::make_shared<agent::DesktopAssetFileReader>(asset_folder_path)),
          player_tank(std::nullptr_t()), player_renderer(std::nullptr_t()),
          wanted_frequency(wanted_frequency), controller_kind(controller_kind), bindings(bindings) {
    }

    void DesktopGameEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
        last_model_matrices_ = model_matrices;

        player_renderer->make_current();
        player_renderer->draw(model_matrices);
    }

    void DesktopGameEnvironment::redraw() const {
        if (!player_renderer || last_model_matrices_.empty()) return;

        player_renderer->make_current();
        player_renderer->draw(last_model_matrices_);
    }

    void DesktopGameEnvironment::resize(const int width, const int height) const {
        if (player_renderer) player_renderer->set_window_size(width, height);
    }

    bool DesktopGameEnvironment::is_player_dead() const {
        const bool is_any_item_dead = std::ranges::any_of(
            player_tank->get_items(), [](const std::shared_ptr<model::Item> &item) {
                if (const auto &life_item = std::dynamic_pointer_cast<model::LifeItem>(item);
                    life_item)
                    return life_item->is_dead();
                return false;
            });

        if (is_any_item_dead) player_tank->destroy();

        return is_any_item_dead;
    }

    int DesktopGameEnvironment::get_score() const { return player_tank->get_score(); }

    std::optional<glm::vec2> DesktopGameEnvironment::aim_point_on_screen() const {
        if (!player_tank || !player_renderer) return std::nullopt;

        const glm::mat4 canon_matrix = player_tank->get_canon()->get_model_matrix();
        const glm::vec3 origin = canon_matrix * glm::vec4(0.f, 0.f, 0.f, 1.f);
        const glm::vec3 forward =
            glm::normalize(glm::mat3(canon_matrix) * glm::vec3(0.f, 0.f, 1.f));
        const glm::vec3 aim = origin + forward * AIM_DISTANCE;

        const glm::vec4 clip = player_renderer->last_view_projection() * glm::vec4(aim, 1.f);
        if (clip.w <= 0.f) return std::nullopt;

        const glm::vec2 ndc = glm::vec2(clip) / clip.w;
        if (ndc.x < -1.f || ndc.x > 1.f || ndc.y < -1.f || ndc.y > 1.f) return std::nullopt;

        return glm::vec2((ndc.x + 1.f) * 0.5f, (1.f - ndc.y) * 0.5f);
    }

    model::PlayerHits DesktopGameEnvironment::consume_player_hits() const {
        return player_tank->consume_hits();
    }

    std::shared_ptr<controller::AbstractKeyboardCallback>
    DesktopGameEnvironment::keyboard_handler() const {
        return keyboard_handler_;
    }

    std::shared_ptr<controller::AbstractGamepadCallback>
    DesktopGameEnvironment::gamepad_handler() const {
        return gamepad_handler_;
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

        if (controller_kind == ControllerKind::Gamepad) {
            const auto player_controller_handler =
                std::make_shared<PlayerGamepadHandler>(bindings.gamepad);

            for (auto &ctrl: player_tank->get_controllers())
                player_controller_handler->add_controller(ctrl);

            gamepad_handler_ = player_controller_handler;
        } else if (controller_kind == ControllerKind::Keyboard) {
            const auto player_controller_handler = std::make_shared<PlayerMouseKeyboardHandler>(
                windowed_backend->get_window(), *player_renderer, bindings.keyboard);

            for (auto &ctrl: player_tank->get_controllers())
                player_controller_handler->add_controller(ctrl);

            keyboard_handler_ = player_controller_handler;
        }

        player_renderer->make_current();

        const auto drawable_factory = windowed_backend->drawable_factory();

        player_renderer->add_drawable(
            "cubemap", drawable_factory->make_cube_map(file_reader, "cubemap/1"));

        std::uniform_real_distribution u_dist(0.f, 1.f);

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

        player_renderer->release_current();
    }

    DesktopGameEnvironment::~DesktopGameEnvironment() {
        std::cout << "Final score : " << player_tank->get_score() << std::endl;
    }

}// namespace arenai::desktop
