//
// Created by claude on 02/11/2026.
//

#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <string>
#include <thread>

#include <argparse/argparse.hpp>
#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include <arenai_model/constants.h>
#include <arenai_view/backend.h>

#include "./agents/ppo_liquid/liquid_hidden_state.h"
#include "./core/train_environment.h"
#include "./distributions/bernoulli.h"
#include "./distributions/truncated_normal.h"
#include "./networks/recurrent/liquid_actor.h"
#include "./networks_utils/torch_converter.h"
#include "./networks_utils/torch_loader.h"

using namespace arenai;
using namespace arenai::agent;

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai probe");

    parser.add_argument("--run_folder").required();
    parser.add_argument("--save").required();
    parser.add_argument("--resources_folder").required();
    parser.add_argument("--output").required();
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(4);
    parser.add_argument("--spawn_side").scan<'g', float>().default_value(550.f);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--deterministic").default_value(false).implicit_value(true);
    parser.add_argument("--vision_num_threads")
        .scan<'i', int>()
        .default_value(static_cast<int>(std::thread::hardware_concurrency()));

    parser.parse_args(argc, argv);

    const std::filesystem::path run_folder(parser.get<std::string>("--run_folder"));
    const std::filesystem::path save_folder = run_folder / parser.get<std::string>("--save");
    const std::filesystem::path resources_folder(parser.get<std::string>("--resources_folder"));
    const std::filesystem::path output(parser.get<std::string>("--output"));
    const int nb_episodes = parser.get<int>("--nb_episodes");
    const float spawn_side = parser.get<float>("--spawn_side");
    const bool deterministic = parser.get<bool>("--deterministic");

    const torch::Device device =
        parser.get<bool>("--cuda") ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU);

    nlohmann::json config;
    std::ifstream(run_folder / "config.json") >> config;
    const auto &agent_config = config.at("agent");
    const auto &env_config = config.at("environment");

    const int vision_height = env_config.at("vision_height").get<int>();
    const int vision_width = env_config.at("vision_width").get<int>();
    const float wanted_frequency = env_config.at("wanted_frequency").get<float>();
    const int nb_tanks = env_config.at("nb_tanks").get<int>();
    const int action_repeat = env_config.at("action_repeat").get<int>();
    const int max_episode_steps = config.at("train").at("max_episode_steps").get<int>();

    const auto actor = std::make_shared<LiquidActor>(
        model::ENEMY_PROPRIOCEPTION_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
        model::ENEMY_NB_DISCRETE_ACTION, agent_config.at("hidden_size_sensors").get<int>(),
        agent_config.at("vision_channels").get<std::vector<std::tuple<int, int>>>(),
        agent_config.at("group_norm_nums").get<std::vector<int>>(),
        agent_config.at("neuron_number").get<int>(), agent_config.at("unfolding_steps").get<int>(),
        agent_config.at("delta_t").get<float>(), agent_config.at("initial_sigma").get<float>(),
        std::vector{
            agent_config.at("initial_fire_proba").get<float>(),
            agent_config.at("initial_zoom_proba").get<float>()});

    load_torch(save_folder, actor, "actor.pt");
    actor->to(device);
    actor->eval();

    LiquidHiddenState hidden_state(actor);

    auto graphics_backend = view::make_vulkan_backend();
    std::cout << "Vulkan device : " << graphics_backend->renderer_info() << std::endl;

    const auto env = std::make_unique<TrainTankEnvironment>(
        std::move(graphics_backend), nb_tanks, resources_folder, wanted_frequency,
        max_episode_steps, vision_height, vision_width, parser.get<int>("--vision_num_threads"),
        action_repeat);

    std::ofstream csv(output);
    csv << "episode;step;tank;distance;yaw;pitch;best_angle;best_distance;can_fire;shells;cooldown;"
           "p_fire;p_zoom;fire;"
           "mu_dir_x;mu_dir_y;mu_canon_x;mu_canon_y;sigma_canon_x;sigma_canon_y;act_canon_x;"
           "act_canon_y\n";

    constexpr float to_degrees = 180.f / std::numbers::pi_v<float>;

    for (int episode = 0; episode < nb_episodes; episode++) {
        hidden_state.reset();

        auto state =
            states_to_tensor(env->reset(spawn_side, spawn_side), vision_height, vision_width);

        int step = 0;
        bool is_done = false;

        while (!is_done) {
            torch::Tensor continuous_action, discrete_action, mu, sigma, discrete;
            {
                torch::NoGradGuard guard;

                const auto x_t = hidden_state.get(state.vision.size(0));
                auto out =
                    actor->act(state.vision.to(device), state.proprioception.to(device), x_t);
                hidden_state.set(out.next_x);

                mu = out.mu.cpu();
                sigma = out.sigma.cpu();
                discrete = out.discrete.cpu();

                if (deterministic) {
                    continuous_action = truncated_normal_mean(mu, sigma);
                    discrete_action = bernoulli_max_action(discrete);
                } else {
                    continuous_action = truncated_normal_sample(mu, sigma);
                    discrete_action = bernoulli_sample(discrete);
                }
            }

            const auto geometry = env->aim_geometry();

            const auto mu_acc = mu.accessor<float, 2>();
            const auto sigma_acc = sigma.accessor<float, 2>();
            const auto discrete_acc = discrete.accessor<float, 2>();
            const auto cont_acc = continuous_action.accessor<float, 2>();
            const auto disc_acc = discrete_action.accessor<float, 2>();
            const auto proprio_acc = state.proprioception.accessor<float, 2>();
            const auto mask_acc = state.fire_mask.accessor<float, 2>();

            for (int i = 0; i < nb_tanks; i++) {
                if (!geometry[i].alive) continue;

                csv << episode << ';' << step << ';' << i << ';' << geometry[i].distance << ';'
                    << geometry[i].yaw * to_degrees << ';' << geometry[i].pitch * to_degrees << ';'
                    << geometry[i].best_angle * to_degrees << ';' << geometry[i].best_distance
                    << ';' << mask_acc[i][0] << ';'
                    << proprio_acc[i][model::ENEMY_PROPRIOCEPTION_SIZE - 2] << ';'
                    << proprio_acc[i][model::ENEMY_PROPRIOCEPTION_SIZE - 1] << ';'
                    << discrete_acc[i][0] << ';' << discrete_acc[i][1] << ';' << disc_acc[i][0]
                    << ';' << mu_acc[i][0] << ';' << mu_acc[i][1] << ';' << mu_acc[i][2] << ';'
                    << mu_acc[i][3] << ';' << sigma_acc[i][2] << ';' << sigma_acc[i][3] << ';'
                    << cont_acc[i][2] << ';' << cont_acc[i][3] << '\n';
            }

            const auto steps =
                env->step(wanted_frequency, tensor_to_actions(continuous_action, discrete_action));

            state = steps_to_tensor(steps, vision_height, vision_width).states;
            is_done = env->is_episode_terminated();
            step++;
        }

        env->stop_drawing();
        std::cout << "episode " << episode << " : " << step << " steps" << std::endl;
    }

    return 0;
}
