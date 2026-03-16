//
// Created by samuel on 11/03/2026.
//

#include "./game_environment.h"

#include <arenai_train/file_reader.h>
#include <arenai_view/cubemap.h>
#include <arenai_view/specular.h>

#include "../model/player_tank_factory.h"
#include "../view/glfw_gl_context.h"

DesktopGameEnvironment::DesktopGameEnvironment(
    const std::shared_ptr<AbstractFileReader> &asset_file_reader,
    const std::shared_ptr<AbstractGLContext> &gl_context, const int window_width,
    const int window_height, const int nb_tanks, const float wanted_frequency)
    : BaseTanksEnvironment(asset_file_reader, gl_context, nb_tanks, wanted_frequency, true),
      asset_file_reader(asset_file_reader), tank_factory(std::nullptr_t()),
      player_renderer(std::nullptr_t()), player_controller_handler(std::nullptr_t()),
      window_width(window_width), window_height(window_height), wanted_frequency(wanted_frequency) {
}

void DesktopGameEnvironment::on_draw(
    const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {
    player_renderer->draw(model_matrices);
}

void DesktopGameEnvironment::on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) {
    tank_factory = std::make_unique<DesktopPlayerTankFactory>(
        asset_file_reader, "player", glm::vec3(0., -40., 40), wanted_frequency);

    for (auto &item: tank_factory->get_items()) { engine->add_item(item); }

    for (auto &item_producer: tank_factory->get_item_producers())
        engine->add_item_producer(item_producer);
}

void DesktopGameEnvironment::on_reset_drawables(
    const std::unique_ptr<PhysicEngine> &engine,
    const std::shared_ptr<AbstractGLContext> &gl_context) {
    player_renderer = std::make_unique<PlayerRenderer>(
        gl_context, window_width, window_height, glm::vec3(200, 300, 200),
        tank_factory->get_camera());
    player_renderer->make_current();

    player_controller_handler = std::make_unique<DesktopPlayerControllerHandler>();

    for (auto &ctrl: tank_factory->get_controllers())
        player_controller_handler->add_controller(ctrl);

    player_renderer->add_drawable("cubemap", std::make_unique<CubeMap>(file_reader, "cubemap/1"));

    std::uniform_real_distribution<float> u_dist(0.f, 1.f);

    for (const auto &[name, shape]: tank_factory->load_shell_shapes()) {
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
    GLFWwindow *window, const int key, const int key_action, const float mouse_x,
    const float mouse_y, const int mouse_button, const int mouse_button_action) const {

    player_controller_handler->on_event(
        {key, key_action, mouse_x, mouse_y, mouse_button, mouse_button_action});
}
