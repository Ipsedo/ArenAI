//
// Created by samuel on 28/09/2025.
//

#include <regex>

#include <argparse/argparse.hpp>

#include "./train.h"
#include "./utils/cli_parser.h"

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai train");

    // model
    parser.add_argument("--vision_channels")
        .default_value<vision_channels>({{{3, 8}, {8, 16}, {16, 32}, {32, 64}, {64, 128}}})
        .action(parse_cli_vision_channels);
    parser.add_argument("--group_norm_nums")
        .default_value<group_norm_nums>({{2, 4, 8, 16, 32}})
        .action(parse_cli_group_norms);
    parser.add_argument("--sensors_hidden_size").scan<'i', int>().default_value(64);
    parser.add_argument("--actions_hidden_size").scan<'i', int>().default_value(32);
    parser.add_argument("--actor_hidden_size")
        .default_value<hidden_layers>({{1024, 512}})
        .action(parse_cli_hidden_layer);
    parser.add_argument("--critic_hidden_size")
        .default_value<hidden_layers>({{1024, 512}})
        .action(parse_cli_hidden_layer);
    parser.add_argument("--tau").scan<'g', float>().default_value(0.005f);
    parser.add_argument("--gamma").scan<'g', float>().default_value(0.99f);
    parser.add_argument("--initial_alpha_continuous").scan<'g', float>().default_value(1e-3f);
    parser.add_argument("--initial_alpha_discrete").scan<'g', float>().default_value(1e-3f);

    // train
    parser.add_argument("--output_folder").required();
    parser.add_argument("--asset_folder").required();
    parser.add_argument("--actor_learning_rate").scan<'g', float>().default_value(1e-4f);
    parser.add_argument("--critic_learning_rate").scan<'g', float>().default_value(3e-4f);
    parser.add_argument("--alpha_learning_rate").scan<'g', float>().default_value(1e-4f);
    parser.add_argument("--potential_reward_scale").scan<'g', float>().default_value(0.1f);
    parser.add_argument("--epochs").scan<'i', int>().default_value(4);
    parser.add_argument("--batch_size").scan<'i', int>().default_value(256);
    parser.add_argument("--max_episode_steps").scan<'i', int>().default_value(30 * 60 * 3);
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(20000);
    parser.add_argument("--replay_buffer_size").scan<'i', int>().default_value(500000);
    parser.add_argument("--train_every").scan<'i', int>().default_value(64);
    parser.add_argument("--save_every").scan<'i', int>().default_value(30 * 60 * 3 * 25);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--metric_window_size").scan<'i', int>().default_value(256);

    // env
    parser.add_argument("--wanted_frequency").scan<'g', float>().default_value(1.f / 30.f);
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(16);
    parser.add_argument("--initial_spawn_width").scan<'g', float>().default_value(500.f);
    parser.add_argument("--initial_spawn_height").scan<'g', float>().default_value(500.f);
    parser.add_argument("--final_spawn_width").scan<'g', float>().default_value(2000.f);
    parser.add_argument("--final_spawn_height").scan<'g', float>().default_value(2000.f);

    parser.parse_args(argc, argv);

    train_main(
        {parser.get<float>("--wanted_frequency"), parser.get<int>("--nb_tanks"),
         parser.get<float>("--initial_spawn_width"), parser.get<float>("--initial_spawn_height"),
         parser.get<float>("--final_spawn_width"), parser.get<float>("--final_spawn_height")},
        {parser.get<vision_channels>("--vision_channels").channels,
         parser.get<group_norm_nums>("--group_norm_nums").groups,
         parser.get<int>("--sensors_hidden_size"), parser.get<int>("--actions_hidden_size"),
         parser.get<hidden_layers>("--actor_hidden_size").layers,
         parser.get<hidden_layers>("--critic_hidden_size").layers, parser.get<float>("--tau"),
         parser.get<float>("--gamma"), parser.get<float>("--initial_alpha_continuous"),
         parser.get<float>("--initial_alpha_discrete")},
        {std::filesystem::path(parser.get<std::string>("--output_folder")),
         std::filesystem::path(parser.get<std::string>("--asset_folder")),
         parser.get<float>("--actor_learning_rate"), parser.get<float>("--critic_learning_rate"),
         parser.get<float>("--alpha_learning_rate"), parser.get<float>("--potential_reward_scale"),
         parser.get<int>("--epochs"), parser.get<int>("--batch_size"),
         parser.get<int>("--max_episode_steps"), parser.get<int>("--nb_episodes"),
         parser.get<int>("--replay_buffer_size"), parser.get<int>("--train_every"),
         parser.get<int>("--save_every"), parser.get<bool>("--cuda"),
         parser.get<int>("--metric_window_size")});

    return 0;
}
