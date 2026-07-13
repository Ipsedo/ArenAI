//
// Created by samuel on 28/09/2025.
//

#include <regex>
#include <thread>

#include <argparse/argparse.hpp>

#include "./train.h"
#include "./utils/cli_parser.h"

using namespace arenai;
using namespace arenai::train;

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai train");

    // model
    parser.add_argument("--vision_channels")
        .default_value<vision_channels>(
            {{{3, 32}, {32, 64}, {64, 128}, {128, 256}, {256, 256}, {256, 256}}})
        .action(parse_cli_vision_channels);
    parser.add_argument("--group_norm_nums")
        .default_value<group_norm_nums>({{4, 8, 16, 32, 32, 32}})
        .action(parse_cli_group_norms);
    parser.add_argument("--sensors_hidden_size").scan<'i', int>().default_value(256);
    parser.add_argument("--actions_hidden_size").scan<'i', int>().default_value(64);
    parser.add_argument("--actor_hidden_sizes")
        .default_value<hidden_layers>({{2560, 1280}})
        .action(parse_cli_hidden_layer);
    parser.add_argument("--critic_hidden_sizes")
        .default_value<hidden_layers>({{2560, 1280}})
        .action(parse_cli_hidden_layer);
    parser.add_argument("--tau").scan<'g', float>().default_value(0.005f);
    parser.add_argument("--gamma").scan<'g', float>().default_value(0.99f);

    // train
    parser.add_argument("--output_folder").required();
    parser.add_argument("--resources_folder_path").required();
    parser.add_argument("--actor_learning_rate").scan<'g', float>().default_value(1e-4f);
    parser.add_argument("--critic_learning_rate").scan<'g', float>().default_value(3e-4f);
    parser.add_argument("--alpha_learning_rate").scan<'g', float>().default_value(3e-4f);
    parser.add_argument("--epochs").scan<'i', int>().default_value(8);
    parser.add_argument("--batch_size").scan<'i', int>().default_value(256);
    parser.add_argument("--max_episode_steps").scan<'i', int>().default_value(30 * 60 * 3);
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(5000);
    parser.add_argument("--replay_buffer_size").scan<'i', int>().default_value(150000);
    parser.add_argument("--train_every").scan<'i', int>().default_value(64);
    parser.add_argument("--save_every").scan<'i', int>().default_value(30 * 60 * 3 * 25);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--metric_window_size").scan<'i', int>().default_value(256);

    // env
    parser.add_argument("--wanted_frequency").scan<'g', float>().default_value(1.f / 30.f);
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(16);
    parser.add_argument("--vision_height").scan<'i', int>().default_value(128);
    parser.add_argument("--vision_width").scan<'i', int>().default_value(256);
    parser.add_argument("--initial_spawn_width").scan<'g', float>().default_value(250.f);
    parser.add_argument("--initial_spawn_height").scan<'g', float>().default_value(250.f);
    parser.add_argument("--final_spawn_width").scan<'g', float>().default_value(1000.f);
    parser.add_argument("--final_spawn_height").scan<'g', float>().default_value(1000.f);
    parser.add_argument("--vision_num_threads")
        .scan<'i', int>()
        .default_value(static_cast<int>(std::thread::hardware_concurrency()));

    parser.parse_args(argc, argv);

    train_main(
        {parser.get<float>("--wanted_frequency"), parser.get<int>("--nb_tanks"),
         parser.get<int>("--vision_height"), parser.get<int>("--vision_width"),
         parser.get<float>("--initial_spawn_width"), parser.get<float>("--initial_spawn_height"),
         parser.get<float>("--final_spawn_width"), parser.get<float>("--final_spawn_height"),
         parser.get<int>("--vision_num_threads")},
        {parser.get<vision_channels>("--vision_channels").channels,
         parser.get<group_norm_nums>("--group_norm_nums").groups,
         parser.get<int>("--sensors_hidden_size"), parser.get<int>("--actions_hidden_size"),
         parser.get<hidden_layers>("--actor_hidden_sizes").layers,
         parser.get<hidden_layers>("--critic_hidden_sizes").layers, parser.get<float>("--tau"),
         parser.get<float>("--gamma")},
        {std::filesystem::path(parser.get<std::string>("--output_folder")),
         std::filesystem::path(parser.get<std::string>("--resources_folder_path")),
         parser.get<float>("--actor_learning_rate"), parser.get<float>("--critic_learning_rate"),
         parser.get<float>("--alpha_learning_rate"), parser.get<int>("--epochs"),
         parser.get<int>("--batch_size"), parser.get<int>("--max_episode_steps"),
         parser.get<int>("--nb_episodes"), parser.get<int>("--replay_buffer_size"),
         parser.get<int>("--train_every"), parser.get<int>("--save_every"),
         parser.get<bool>("--cuda"), parser.get<int>("--metric_window_size")});

    return 0;
}
