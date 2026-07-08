//
// Created by samuel on 18/03/2026.
//

#include "./game.h"

#include <iostream>

#include <torch/torch.h>

#include <arenai_core/constants.h>
#include <arenai_train/factory_set.h>
#include <arenai_train/torch_converter.h>
#include <arenai_view/factory.h>

#include "./core/game_environment.h"

using namespace arenai;

namespace arenai::desktop {

    void game_loop(const GameOptions &game_options, const ModelOptions &model_options) {
        torch::NoGradGuard no_grad;

        torch::Device device = model_options.cuda ? torch::kCUDA : torch::kCPU;

        const auto sac_agent =
            train::SacAgentFactory(model_options.hyper_parameters)
                .get_agent(
                    model_options.vision_height, model_options.vision_height,
                    model::ENEMY_PROPRIOCEPTION_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
                    model::ENEMY_NB_DISCRETE_ACTION);
        sac_agent->set_train(false);

        sac_agent->load(std::filesystem::path(model_options.state_dict_folder));
        sac_agent->to(device);

        std::cout << "Parameters : " << sac_agent->count_parameters() << std::endl;

        // The view owns the window + GL context; the app only speaks the abstract
        // window/backend interface.
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> graphics_backend =
            view::make_glfw_backend(
                game_options.window_width, game_options.window_height, "ArenAI");
        const auto window = graphics_backend->get_window();

        const auto env = std::make_shared<DesktopGameEnvironment>(
            game_options.android_asset_folder, graphics_backend, game_options.nb_tanks,
            model_options.vision_height, model_options.vision_width, game_options.wanted_frequency);

        auto states = env->reset(500, 500);

        std::cout << "OpenGL : " << graphics_backend->renderer_info() << std::endl;

        const auto frame_dt =
            std::chrono::milliseconds(static_cast<int>(game_options.wanted_frequency * 1000.f));

        while (!window->should_close()) {
            window->poll_events();

            auto last_time = std::chrono::steady_clock::now();

            const auto [vision, proprioception] = train::states_to_tensor(
                states, model_options.vision_height, model_options.vision_width);

            const auto [continuous_action, discrete_action] =
                sac_agent->act(vision.to(device), proprioception.to(device));
            const auto actions_for_env =
                train::tensor_to_actions(continuous_action.cpu(), discrete_action.cpu());

            const auto steps = env->step(game_options.wanted_frequency, actions_for_env);

            states.clear();

            for (const auto &[state, reward, done, is_truncated]: steps) states.push_back(state);

            auto now = std::chrono::steady_clock::now();
            auto dt = now - last_time;

            std::this_thread::sleep_for(
                std::max(frame_dt - dt, std::chrono::steady_clock::duration::zero()));
        }
    }

}// namespace arenai::desktop
