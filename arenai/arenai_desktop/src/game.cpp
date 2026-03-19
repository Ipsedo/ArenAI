//
// Created by samuel on 18/03/2026.
//

#include "./game.h"

#include <GLFW/glfw3.h>
#include <torch/torch.h>

#include <arenai_core/constants.h>
#include <arenai_train/factory_set.h>
#include <arenai_train/torch_converter.h>

#include "./core/game_environment.h"

void game_loop(const GameOptions &game_options, const ModelOptions &model_options) {
    torch::NoGradGuard no_grad;

    torch::Device device = model_options.cuda ? torch::kCUDA : torch::kCPU;

    const auto sac_agent =
        SacAgentFactory(model_options.hyper_parameters)
            .get_agent(
                ENEMY_VISION_HEIGHT, ENEMY_VISION_WIDTH, ENEMY_PROPRIOCEPTION_SIZE,
                ENEMY_NB_CONTINUOUS_ACTION, ENEMY_NB_DISCRETE_ACTION);
    sac_agent->set_train(false);

    sac_agent->load(std::filesystem::path(model_options.state_dict_folder));
    sac_agent->to(device);

    std::cout << "Parameters : " << sac_agent->count_parameters() << std::endl;

    if (!glfwInit()) throw std::runtime_error("glfwInit() failed");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_EGL_CONTEXT_API);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    auto glfw_window = glfwCreateWindow(
        game_options.window_width, game_options.window_height, "ArenAI", nullptr, nullptr);

    const auto env = std::make_shared<DesktopGameEnvironment>(
        game_options.android_asset_folder, glfw_window, game_options.nb_tanks,
        game_options.wanted_frequency);

    auto states = env->reset_physics();
    env->reset_drawables();

    const auto frame_dt =
        std::chrono::milliseconds(static_cast<int>(game_options.wanted_frequency * 1000.f));

    while (!glfwWindowShouldClose(glfw_window)) {
        glfwPollEvents();

        auto last_time = std::chrono::steady_clock::now();

        const auto [vision, proprioception] = states_to_tensor(states);

        const auto [continuous_action, discrete_action] =
            sac_agent->act(vision.to(device), proprioception.to(device));
        const auto actions_for_env =
            tensor_to_actions(continuous_action.cpu(), discrete_action.cpu());

        const auto steps = env->step(game_options.wanted_frequency, actions_for_env);

        states.clear();

        for (const auto &[state, reward, done]: steps) states.push_back(state);

        auto now = std::chrono::steady_clock::now();
        auto dt = now - last_time;

        std::this_thread::sleep_for(
            std::max(frame_dt - dt, std::chrono::steady_clock::duration::zero()));
    }
}
