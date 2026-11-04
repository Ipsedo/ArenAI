//
// Created by claude on 02/11/2026.
//

#include <cmath>
#include <deque>
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
#include <arenai_model/shell_economy.h>
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
    parser.add_argument("--shells")
        .scan<'i', int>()
        .default_value(model::GAME_SHELL_ECONOMY.initial_shells);
    parser.add_argument("--shells_per_hit")
        .scan<'i', int>()
        .default_value(model::GAME_SHELL_ECONOMY.shells_per_hit);
    parser.add_argument("--max_shells")
        .scan<'i', int>()
        .default_value(model::GAME_SHELL_ECONOMY.max_shells);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--deterministic").default_value(false).implicit_value(true);
    parser.add_argument("--features").default_value(std::string());
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
    const model::ShellEconomy shell_economy{
        .initial_shells = parser.get<int>("--shells"),
        .shells_per_hit = parser.get<int>("--shells_per_hit"),
        .max_shells = parser.get<int>("--max_shells")};
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

    std::ofstream shots_csv(output.parent_path() / (output.stem().string() + "_shots.csv"));
    shots_csv
        << "episode;tank;fire_step;land_step;fire_best_angle;fire_best_distance;fire_distance;"
           "fire_yaw;fire_pitch;shells_at_fire;group_size;group_hits;group_miss\n";

    const auto features_path = parser.get<std::string>("--features");
    std::ofstream features_out;
    if (!features_path.empty()) features_out.open(features_path, std::ios::binary);

    struct PendingShot {
        int step;
        AimGeometry geometry;
        float shells;
    };

    constexpr float to_degrees = 180.f / std::numbers::pi_v<float>;

    for (int episode = 0; episode < nb_episodes; episode++) {
        hidden_state.reset();

        auto state = states_to_tensor(
            env->reset(spawn_side, spawn_side, shell_economy), vision_height, vision_width);

        int step = 0;
        bool is_done = false;
        std::vector<std::deque<PendingShot>> pending(nb_tanks);

        while (!is_done) {
            torch::Tensor continuous_action, discrete_action, mu, sigma, discrete, features,
                vision_features;
            {
                torch::NoGradGuard guard;

                const auto x_t = hidden_state.get(state.vision.size(0));
                auto out =
                    actor->act(state.vision.to(device), state.proprioception.to(device), x_t);
                hidden_state.set(out.next_x);

                mu = out.mu.cpu();
                sigma = out.sigma.cpu();
                discrete = out.discrete.cpu();
                features = out.features.cpu().contiguous();
                vision_features = out.vision_features.cpu().contiguous();

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

                if (features_out.is_open()) {
                    const float header[3] = {
                        static_cast<float>(episode), static_cast<float>(step),
                        static_cast<float>(i)};
                    features_out.write(reinterpret_cast<const char *>(header), sizeof(header));
                    features_out.write(
                        reinterpret_cast<const char *>(features[i].data_ptr<float>()),
                        features.size(1) * sizeof(float));
                    features_out.write(
                        reinterpret_cast<const char *>(vision_features[i].data_ptr<float>()),
                        vision_features.size(1) * sizeof(float));
                }

                if (mask_acc[i][0] > 0.5f && disc_acc[i][0] > 0.5f)
                    pending[i].push_back(
                        {step, geometry[i], proprio_acc[i][model::ENEMY_PROPRIOCEPTION_SIZE - 2]});
            }

            const auto steps =
                env->step(wanted_frequency, tensor_to_actions(continuous_action, discrete_action));

            state = steps_to_tensor(steps, vision_height, vision_width).states;

            const auto details = env->last_step_reward_details();
            for (int i = 0; i < nb_tanks; i++) {
                const int landed = details[i].nb_landed_shells;
                if (landed == 0) continue;

                const float hit_sum = details[i].hit;
                const int kills = static_cast<int>(hit_sum / 2.f);
                const int hits =
                    kills + static_cast<int>(std::round((hit_sum - 2.f * kills) / 0.3f));
                const float miss = details[i].sum_miss_distance / static_cast<float>(landed);

                for (int k = 0; k < landed && !pending[i].empty(); k++) {
                    const auto &shot = pending[i].front();
                    shots_csv << episode << ';' << i << ';' << shot.step << ';' << step << ';'
                              << shot.geometry.best_angle * to_degrees << ';'
                              << shot.geometry.best_distance << ';' << shot.geometry.distance << ';'
                              << shot.geometry.yaw * to_degrees << ';'
                              << shot.geometry.pitch * to_degrees << ';' << shot.shells << ';'
                              << landed << ';' << hits << ';' << miss << '\n';
                    pending[i].pop_front();
                }
            }
            is_done = env->is_episode_terminated();
            step++;
        }

        env->stop_drawing();
        std::cout << "episode " << episode << " : " << step << " steps" << std::endl;
    }

    return 0;
}
