//
// Created by samuel on 28/09/2025.
//

#include <argparse/argparse.hpp>

#include "./train.h"

int main(int argc, char **argv) {

    argparse::ArgumentParser parser("arenai train");

    // model
    parser.add_argument("--hidden_size_sensors").scan<'i', int>().default_value(128);
    parser.add_argument("--hidden_size_actions").scan<'i', int>().default_value(64);
    parser.add_argument("--actor_hidden_size").scan<'i', int>().default_value(1024);
    parser.add_argument("--critic_hidden_size").scan<'i', int>().default_value(1024);
    parser.add_argument("--tau").scan<'g', float>().default_value(0.005f);
    parser.add_argument("--gamma").scan<'g', float>().default_value(0.99f);
    parser.add_argument("--initial_alpha").scan<'g', float>().default_value(1.f);

    // train
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(8);
    parser.add_argument("--output_folder").required();
    parser.add_argument("--asset_folder").required();
    parser.add_argument("--potential_reward_scale").scan<'g', float>().default_value(10.f);
    parser.add_argument("--learning_rate").scan<'g', float>().default_value(3e-4f);
    parser.add_argument("--epochs").scan<'i', int>().default_value(5);
    parser.add_argument("--batch_size").scan<'i', int>().default_value(400);
    parser.add_argument("--max_episode_steps").scan<'i', int>().default_value(30 * 60);
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(30000);
    parser.add_argument("--replay_buffer_size").scan<'i', int>().default_value(200000);
    parser.add_argument("--train_every").scan<'i', int>().default_value(125);
    parser.add_argument("--save_every").scan<'i', int>().default_value(30 * 60 * 3 * 25);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--metric_window_size").scan<'i', int>().default_value(10000);

    parser.parse_args(argc, argv);

    train_main(
        {parser.get<int>("--hidden_size_sensors"), parser.get<int>("--hidden_size_actions"),
         parser.get<int>("--actor_hidden_size"), parser.get<int>("--critic_hidden_size"),
         parser.get<float>("--tau"), parser.get<float>("--gamma"),
         parser.get<float>("--initial_alpha")},
        {parser.get<int>("--nb_tanks"),
         std::filesystem::path(parser.get<std::string>("--output_folder")),
         std::filesystem::path(parser.get<std::string>("--asset_folder")),
         parser.get<float>("--potential_reward_scale"), parser.get<float>("--learning_rate"),
         parser.get<int>("--epochs"), parser.get<int>("--batch_size"),
         parser.get<int>("--max_episode_steps"), parser.get<int>("--nb_episodes"),
         parser.get<int>("--replay_buffer_size"), parser.get<int>("--train_every"),
         parser.get<int>("--save_every"), parser.get<bool>("--cuda"),
         parser.get<int>("--metric_window_size")});

    return 0;
}
