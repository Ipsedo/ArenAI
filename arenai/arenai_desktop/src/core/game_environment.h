//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_ENVIRONMENT_H
#define ARENAI_DESKTOP_GAME_ENVIRONMENT_H

#include <GLFW/glfw3.h>

#include <arenai_core/environment.h>
#include <arenai_core/player_tank_factory.h>
#include <arenai_view/renderer.h>

#include "../controller/player_controller_handler.h"

class DesktopGameEnvironment : public BaseTanksEnvironment {
public:
    DesktopGameEnvironment(
        const std::filesystem::path &asset_folder_path, GLFWwindow *glfw_window, int nb_tanks,
        float wanted_frequency);

    ~DesktopGameEnvironment() override;

protected:
    void on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

    void on_reset_physics(const std::unique_ptr<PhysicEngine> &engine) override;

    void on_reset_drawables(
        const std::unique_ptr<PhysicEngine> &engine,
        const std::shared_ptr<AbstractGLContext> &gl_context) override;

private:
    GLFWwindow *curr_window;

    std::shared_ptr<AbstractFileReader> asset_file_reader;
    std::unique_ptr<PlayerTankFactory> player_tank_factory;
    std::unique_ptr<PlayerRenderer> player_renderer;
    std::unique_ptr<MouseKeyboardPlayerControllerHandler> player_controller_handler;

    int window_width;
    int window_height;

    float wanted_frequency;

    void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) const;
    void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) const;
    void cursor_position_callback(GLFWwindow *window, double xpos, double ypos) const;

    void global_glfw_callback(
        GLFWwindow *window, int key, int key_action, double mouse_x, double mouse_y,
        int mouse_button, int mouse_button_action) const;
};

#endif//ARENAI_DESKTOP_GAME_ENVIRONMENT_H
