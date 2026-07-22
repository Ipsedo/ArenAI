//
// Created by samuel on 28/09/2025.
//

#include <thread>

#include <argparse/argparse.hpp>

#include "./agents/agent_cli.h"
#include "./train.h"

using namespace arenai;
using namespace arenai::agent;

int main(const int argc, char **argv) {
    // declared before the parser: add_subparser keeps references on the subparsers
    const auto algorithms = make_agent_clis();

    argparse::ArgumentParser parser("arenai train");

    // train
    parser.add_group("train");
    parser.add_argument("--output_folder").required();
    parser.add_argument("--resources_folder").required();
    parser.add_argument("--max_episode_steps").scan<'i', int>().default_value(30 * 60 * 3);
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(1000);
    parser.add_argument("--save_every").scan<'i', int>().default_value(30 * 60 * 3 * 5);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);

    // env
    parser.add_group("environment");
    parser.add_argument("--wanted_frequency").scan<'g', float>().default_value(1.f / 30.f);
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(32);
    parser.add_argument("--vision_height").scan<'i', int>().default_value(128);
    parser.add_argument("--vision_width").scan<'i', int>().default_value(256);
    parser.add_argument("--initial_spawn_width").scan<'g', float>().default_value(500.f);
    parser.add_argument("--initial_spawn_height").scan<'g', float>().default_value(500.f);
    parser.add_argument("--final_spawn_width").scan<'g', float>().default_value(2000.f);
    parser.add_argument("--final_spawn_height").scan<'g', float>().default_value(2000.f);
    parser.add_argument("--vision_num_threads")
        .scan<'i', int>()
        .default_value(static_cast<int>(std::thread::hardware_concurrency()));

    // one subcommand per algorithm, carrying its own hyper-parameter options
    for (const auto &algorithm: algorithms) parser.add_subparser(*algorithm.parser);

    parser.parse_args(argc, argv);

    const AgentCli *selected_algorithm = nullptr;
    for (const auto &algorithm: algorithms)
        if (parser.is_subcommand_used(algorithm.name)) selected_algorithm = &algorithm;

    if (selected_algorithm == nullptr)
        // no subcommand given: default algorithm with its default hyper-parameters
        for (const auto &algorithm: algorithms)
            if (algorithm.name == DEFAULT_AGENT) {
                algorithm.parser->parse_args({DEFAULT_AGENT});
                selected_algorithm = &algorithm;
            }

    const bool cuda = parser.get<bool>("--cuda");
    const int vision_height = parser.get<int>("--vision_height");
    const int vision_width = parser.get<int>("--vision_width");

    const auto agent_factory = selected_algorithm->create_factory(
        vision_height, vision_width,
        cuda ? torch::Device(torch::kCUDA) : torch::Device(torch::kCPU));

    train_main(
        {parser.get<float>("--wanted_frequency"), parser.get<int>("--nb_tanks"), vision_height,
         vision_width, parser.get<float>("--initial_spawn_width"),
         parser.get<float>("--initial_spawn_height"), parser.get<float>("--final_spawn_width"),
         parser.get<float>("--final_spawn_height"), parser.get<int>("--vision_num_threads")},
        {std::filesystem::path(parser.get<std::string>("--output_folder")),
         std::filesystem::path(parser.get<std::string>("--resources_folder")),
         parser.get<int>("--max_episode_steps"), parser.get<int>("--nb_episodes"),
         parser.get<int>("--save_every"), cuda},
        agent_factory);

    return 0;
}
