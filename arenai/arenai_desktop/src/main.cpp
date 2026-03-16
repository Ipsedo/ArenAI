//
// Created by samuel on 11/03/2026.
//

#include <arenai_core/constants.h>
#include <arenai_train/factory_set.h>
#include <arenai_train/file_reader.h>
#include <arenai_train/torch_converter.h>

#include "./core/game_environment.h"
#include "./view/glfw_gl_context.h"

int main(int argc, char **argv) {

    torch::NoGradGuard no_grad;

    const auto sac_agent = SacAgentFactory({}).get_agent(
        ENEMY_VISION_HEIGHT, ENEMY_VISION_WIDTH, ENEMY_PROPRIOCEPTION_SIZE,
        ENEMY_NB_CONTINUOUS_ACTION, ENEMY_NB_DISCRETE_ACTION);
    sac_agent->set_train(false);

    sac_agent->load(std::filesystem::path("/home/samuel/Téléchargements/arenai_save_5/save_5"));
    sac_agent->to(torch::kCPU);

    std::cout << "Parameters : " << sac_agent->count_parameters() << std::endl;

    float wanted_frequency = 1.f / 30.f;
    int window_width = 1920, window_height = 1080;

    if (!glfwInit()) throw std::runtime_error("glfwInit() failed");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    auto glfw_window = glfwCreateWindow(window_width, window_height, "ArenAI", nullptr, nullptr);

    auto gl_context = std::make_shared<GlfwGlContext>(glfw_window);

    auto asset_file_reader = std::make_shared<DesktopAssetFileReader>(
        "/home/samuel/StudioProjects/PhyVR/app/src/main/assets");
    const auto env = std::make_shared<DesktopGameEnvironment>(
        asset_file_reader, gl_context, window_width, window_height, 4, wanted_frequency);

    glfwSetWindowUserPointer(glfw_window, env.get());

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

    auto states = env->reset_physics();
    env->reset_drawables(gl_context);

    const auto frame_dt = std::chrono::milliseconds(static_cast<int>(wanted_frequency * 1000.f));

    while (!gl_context->should_close_window()) {
        glfwPollEvents();

        auto last_time = std::chrono::steady_clock::now();

        const auto [vision, proprioception] = states_to_tensor(states);

        const auto [continuous_action, discrete_action] = sac_agent->act(vision, proprioception);
        const auto actions_for_env = tensor_to_actions(continuous_action, discrete_action);

        const auto steps = env->step(wanted_frequency, actions_for_env);

        states.clear();

        for (const auto [state, reward, done]: steps) states.push_back(state);

        auto now = std::chrono::steady_clock::now();
        auto dt = now - last_time;

        std::this_thread::sleep_for(
            std::max(frame_dt - dt, std::chrono::steady_clock::duration::zero()));
    }

    return 0;
}
