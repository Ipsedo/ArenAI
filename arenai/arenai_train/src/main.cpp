//
// Created by samuel on 28/09/2025.
//

#include <regex>

#include <argparse/argparse.hpp>

#include "./train.h"

struct vision_channels {
    std::vector<std::tuple<int, int>> channels;
};

struct group_norm_nums {
    std::vector<int> groups;
};

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai train");

    // physic simulation
    parser.add_argument("--wanted_frequency").scan<'g', float>().default_value(1.f / 30.f);

    // model
    parser.add_argument("--vision_channels")
        .default_value<vision_channels>(
            {{{3, 8}, {8, 16}, {16, 32}, {32, 48}, {48, 64}, {64, 128}}})
        .action([](const std::string &value) -> vision_channels {
            const std::regex regex_match(
                R"(^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$)");
            const std::regex regex_layer(R"(\( *\d+ *, *\d+ *\))");
            const std::regex regex_channel(R"(\d+)");

            if (!std::regex_match(value.begin(), value.end(), regex_match))
                throw std::invalid_argument(
                    "invalid --vision_channels format, usage : [(10, 20), (20, 40), ...]");

            vision_channels vision_channels;

            std::sregex_iterator it_layer(value.begin(), value.end(), regex_layer);
            for (const std::sregex_iterator end; it_layer != end; ++it_layer) {
                const auto layer_str = it_layer->str();

                const std::sregex_iterator it_channel(
                    layer_str.begin(), layer_str.end(), regex_channel);

                const int c_i = std::stoi(it_channel->str());
                const int c_o = std::stoi(std::next(it_channel)->str());

                vision_channels.channels.emplace_back(c_i, c_o);
            }

            return vision_channels;
        });
    parser.add_argument("--group_norm_nums")
        .default_value<group_norm_nums>({{2, 4, 8, 12, 16, 32}})
        .action([](const std::string &value) -> group_norm_nums {
            const std::regex regex_match(R"(^ *\[(?: *\d+ *,)* *\d+ *] *$)");
            const std::regex regex_groups(R"(\d+)");

            if (!std::regex_match(value.begin(), value.end(), regex_match))
                throw std::invalid_argument(
                    "invalid --group_norm_nums format, usage : [4, 8, 16, ...]");

            group_norm_nums group_nums;

            std::sregex_iterator it_layer(value.begin(), value.end(), regex_groups);
            for (const std::sregex_iterator end; it_layer != end; ++it_layer)
                group_nums.groups.emplace_back(std::stoi(it_layer->str()));

            return group_nums;
        });
    parser.add_argument("--sensors_hidden_size").scan<'i', int>().default_value(256);
    parser.add_argument("--actions_hidden_size").scan<'i', int>().default_value(64);
    parser.add_argument("--actor_hidden_size").scan<'i', int>().default_value(1024);
    parser.add_argument("--critic_hidden_size").scan<'i', int>().default_value(1024);
    parser.add_argument("--tau").scan<'g', float>().default_value(0.005f);
    parser.add_argument("--gamma").scan<'g', float>().default_value(0.99f);
    parser.add_argument("--initial_alpha").scan<'g', float>().default_value(1.f);

    // train
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(8);
    parser.add_argument("--output_folder").required();
    parser.add_argument("--asset_folder").required();
    parser.add_argument("--learning_rate").scan<'g', float>().default_value(3e-4f);
    parser.add_argument("--epochs").scan<'i', int>().default_value(16);
    parser.add_argument("--batch_size").scan<'i', int>().default_value(256);
    parser.add_argument("--max_episode_steps").scan<'i', int>().default_value(30 * 60 * 3);
    parser.add_argument("--nb_episodes").scan<'i', int>().default_value(50000);
    parser.add_argument("--replay_buffer_size").scan<'i', int>().default_value(100000);
    parser.add_argument("--train_every").scan<'i', int>().default_value(256);
    parser.add_argument("--save_every").scan<'i', int>().default_value(30 * 60 * 3 * 25);
    parser.add_argument("--cuda").default_value(false).implicit_value(true);
    parser.add_argument("--metric_window_size").scan<'i', int>().default_value(256);

    parser.parse_args(argc, argv);

    train_main(
        parser.get<float>("--wanted_frequency"),
        {parser.get<vision_channels>("--vision_channels").channels,
         parser.get<group_norm_nums>("--group_norm_nums").groups,
         parser.get<int>("--sensors_hidden_size"), parser.get<int>("--actions_hidden_size"),
         parser.get<int>("--actor_hidden_size"), parser.get<int>("--critic_hidden_size"),
         parser.get<float>("--tau"), parser.get<float>("--gamma"),
         parser.get<float>("--initial_alpha")},
        {parser.get<int>("--nb_tanks"),
         std::filesystem::path(parser.get<std::string>("--output_folder")),
         std::filesystem::path(parser.get<std::string>("--asset_folder")),
         parser.get<float>("--learning_rate"), parser.get<int>("--epochs"),
         parser.get<int>("--batch_size"), parser.get<int>("--max_episode_steps"),
         parser.get<int>("--nb_episodes"), parser.get<int>("--replay_buffer_size"),
         parser.get<int>("--train_every"), parser.get<int>("--save_every"),
         parser.get<bool>("--cuda"), parser.get<int>("--metric_window_size")});

    return 0;
}
