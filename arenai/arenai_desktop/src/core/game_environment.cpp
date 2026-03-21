//
// Created by samuel on 11/03/2026.
//

#include "./game_environment.h"

#include <iostream>

#include <arenai_core/player_tank_factory.h>
#include <arenai_train/file_reader.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/specular.h>

#include "../view/glfw_gl_context.h"

DesktopGameEnvironment::DesktopGameEnvironment(
    const std::string &asset_folder_path, GLFWwindow *glfw_window, const int nb_tanks,
    const float wanted_frequency)
    : BaseTanksEnvironment(
        std::make_shared<DesktopAssetFileReader>(asset_folder_path),
        std::make_shared<GlfwGlContext>(glfw_window), nb_tanks, wanted_frequency, true),
      curr_window(glfw_window),
      asset_file_reader(std::make_shared<DesktopAssetFileReader>(asset_folder_path)),
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

void DesktopGameEnvironment::on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) {
    player_tank_factory = std::make_unique<PlayerTankFactory>(
        asset_file_reader, "player", glm::vec3(0., -40., 40), wanted_frequency);

    for (auto &item: player_tank_factory->get_items()) { engine->add_item(item); }

    for (auto &item_producer: player_tank_factory->get_item_producers())
        engine->add_item_producer(item_producer);
}

void DesktopGameEnvironment::on_reset_drawables(
    const std::unique_ptr<PhysicEngine> &engine,
    const std::shared_ptr<AbstractGLContext> &gl_context) {
    player_renderer = std::make_unique<PlayerRenderer>(
        gl_context, window_width, window_height, glm::vec3(200, 300, 200),
        player_tank_factory->get_camera());
    player_renderer->make_current();

    player_controller_handler = std::make_unique<MouseKeyboardPlayerControllerHandler>(curr_window);

    for (auto &ctrl: player_tank_factory->get_controllers())
        player_controller_handler->add_controller(ctrl);

    player_renderer->add_drawable("cubemap", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    std::uniform_real_distribution<float> u_dist(0.f, 1.f);

    for (const auto &[name, shape]: player_tank_factory->load_shell_shapes()) {
        glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

        player_renderer->add_drawable(
            name, std::make_unique<Specular>(
                      file_reader, shape->get_vertices(), shape->get_normals(), color, color, color,
                      50.f, shape->get_id()));
    }

    for (const auto &item: engine->get_items()) {
        glm::vec4 color(u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, u_dist(rng) * 0.8f, 1.f);

        player_renderer->add_drawable(
            item->get_name(),
            std::make_unique<Specular>(
                file_reader, item->get_shape()->get_vertices(), item->get_shape()->get_normals(),
                color, color, color, 50.f, item->get_shape()->get_id()));
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
        {key, key_action, static_cast<float>(mouse_x), static_cast<float>(mouse_y), mouse_button,
         mouse_button_action});
}

std::vector<std::tuple<State, Reward, IsDone>>
DesktopGameEnvironment::step(const float time_delta, const std::vector<Action> &actions) {
    double x_pos = 0, y_pos = 0;
    glfwGetCursorPos(curr_window, &x_pos, &y_pos);

    global_glfw_callback(curr_window, -1, -1, x_pos, y_pos, -1, -1);

    return BaseTanksEnvironment::step(time_delta, actions);
}

DesktopGameEnvironment::~DesktopGameEnvironment() {
    std::cout << "Final score : " << player_tank_factory->get_score() << std::endl;
}
