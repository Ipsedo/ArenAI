//
// Created by samuel on 11/03/2026.
//

#include "./game_environment.h"

#include <iostream>

#include <arenai_model/engine.h>
#include <arenai_model/tank.h>
#include <arenai_model/tank_factory.h>
#include <arenai_train/file_reader.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/specular.h>

#include "../view/glfw_gl_context.h"

using namespace arenai;

namespace arenai::desktop {

    DesktopGameEnvironment::DesktopGameEnvironment(
        const std::filesystem::path &asset_folder_path, GLFWwindow *glfw_window, const int nb_tanks,
        const int vision_height, const int vision_width, const float wanted_frequency)
        : core::BaseTanksEnvironment(
            std::make_shared<train::DesktopAssetFileReader>(asset_folder_path),
            std::make_shared<GlfwGlContext>(glfw_window), nb_tanks, wanted_frequency, vision_height,
            vision_width, 8, true),
          curr_window(glfw_window),
          asset_file_reader(std::make_shared<train::DesktopAssetFileReader>(asset_folder_path)),
          player_tank_factory(std::nullptr_t()), player_renderer(std::nullptr_t()),
          player_controller_handler(std::nullptr_t()), window_width(0.f), window_height(0.f),
          wanted_frequency(wanted_frequency) {

        glfwGetWindowSize(glfw_window, &window_width, &window_height);

        // callback
        glfwSetWindowUserPointer(glfw_window, this);

        glfwSetKeyCallback(
            glfw_window,
            [](GLFWwindow *window, const int key, const int scancode, const int action,
               const int mods) -> void {
                const auto curr_env =
                    static_cast<DesktopGameEnvironment *>(glfwGetWindowUserPointer(window));
                curr_env->key_callback(window, key, scancode, action, mods);
            });

        glfwSetCursorPosCallback(
            glfw_window, [](GLFWwindow *window, const double xpos, const double ypos) -> void {
                const auto curr_env =
                    static_cast<DesktopGameEnvironment *>(glfwGetWindowUserPointer(window));
                curr_env->cursor_position_callback(window, xpos, ypos);
            });

        glfwSetMouseButtonCallback(
            glfw_window,
            [](GLFWwindow *window, const int button, const int action, const int mods) -> void {
                const auto curr_env =
                    static_cast<DesktopGameEnvironment *>(glfwGetWindowUserPointer(window));
                curr_env->mouse_button_callback(window, button, action, mods);
            });
    }

    void DesktopGameEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
        player_renderer->draw(model_matrices);
    }

    void DesktopGameEnvironment::on_reset_physics(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {
        auto tank_factory = model::make_tank_factory(*engine, asset_file_reader, wanted_frequency);
        player_tank_factory = tank_factory->make_player_tank("player", glm::vec3(0., -40., 40));

        player_controller_handler =
            std::make_unique<MouseKeyboardPlayerControllerHandler>(curr_window);

        for (auto &ctrl: player_tank_factory->get_controllers())
            player_controller_handler->add_controller(ctrl);
    }

    void DesktopGameEnvironment::on_reset_drawables(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine,
        const std::shared_ptr<view::AbstractGLContext> &gl_context) {
        player_renderer = std::make_unique<view::PlayerRenderer>(
            gl_context, window_width, window_height, glm::vec3(200, 300, 200),
            player_tank_factory->get_camera());

        player_renderer->make_current();

        player_renderer->add_drawable(
            "cubemap", std::make_unique<view::CubeMap>(file_reader, "cubemap/1"));

        std::uniform_real_distribution<float> u_dist(0.f, 1.f);

        for (const auto &[name, shape]: player_tank_factory->load_shell_shapes()) {
            glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

            player_renderer->add_drawable(
                name, std::make_unique<view::Specular>(
                          file_reader, shape->get_vertices(), shape->get_normals(), color, color,
                          color, 50.f));
        }

        for (const auto &item: engine->get_items()) {
            glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

            player_renderer->add_drawable(
                item->get_name(), std::make_unique<view::Specular>(
                                      file_reader, item->get_shape()->get_vertices(),
                                      item->get_shape()->get_normals(), color, color, color, 50.f));
        }

        /*for (auto &hud_drawable: player_controller_handler->get_hud_drawables(file_reader))
        player_renderer->add_hud_drawable(std::move(hud_drawable));*/

        player_renderer->release_current();
    }

    void DesktopGameEnvironment::key_callback(
        GLFWwindow *window, const int key, int scancode, const int action, int mods) const {
        double x_pos = 0, y_pos = 0;
        glfwGetCursorPos(window, &x_pos, &y_pos);

        global_glfw_callback(window, key, action, x_pos, y_pos, -1, -1);
    }

    void DesktopGameEnvironment::mouse_button_callback(
        GLFWwindow *window, const int button, const int action, int mods) const {
        double x_pos = 0, y_pos = 0;
        glfwGetCursorPos(window, &x_pos, &y_pos);

        global_glfw_callback(window, -1, -1, x_pos, y_pos, button, action);
    }

    void DesktopGameEnvironment::cursor_position_callback(
        GLFWwindow *window, const double xpos, const double ypos) const {

        global_glfw_callback(window, -1, -1, xpos, ypos, -1, -1);
    }

    void DesktopGameEnvironment::global_glfw_callback(
        GLFWwindow *window, const int key, const int key_action, const double mouse_x,
        const double mouse_y, const int mouse_button, const int mouse_button_action) const {

        player_controller_handler->on_event(
            {key, key_action, static_cast<float>(mouse_x), static_cast<float>(mouse_y),
             mouse_button, mouse_button_action});
    }

    DesktopGameEnvironment::~DesktopGameEnvironment() {
        std::cout << "Final score : " << player_tank_factory->get_score() << std::endl;
    }

}// namespace arenai::desktop
