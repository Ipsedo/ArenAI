//
// Created by samuel on 11/03/2026.
//

#include <regex>

#include <argparse/argparse.hpp>

#include "./game.h"

using namespace arenai;
using namespace arenai::desktop;

void trim_inplace(std::string &s) {
    auto not_space = [](const unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::ranges::find_if(s, not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
}

std::tuple<std::string, std::string> parse_key_value(const std::string &value) {
    const std::regex regex_match(R"(^ *([^=]+)=\"?([^"]+)\"? *$)");

    std::smatch match;

    if (!std::regex_match(value, match, regex_match))
        throw std::invalid_argument("invalid pairs format, usage : key=value");

    std::string s1 = match[1].str();
    std::string s2 = match[2].str();
    trim_inplace(s2);

    return {s1, s2};
}

typedef std::vector<std::tuple<std::string, std::string>> hyper_params_vector;

int main(const int argc, char **argv) {
    argparse::ArgumentParser parser("arenai game");

    // Game options
    parser.add_argument("--wanted_frequency").scan<'g', float>().default_value(1.f / 30.f);
    parser.add_argument("--nb_tanks").scan<'i', int>().default_value(16);
    parser.add_argument("--vision_height").scan<'i', int>().default_value(128);
    parser.add_argument("--vision_width").scan<'i', int>().default_value(256);
    parser.add_argument("--window_width").scan<'i', int>().default_value(1920);
    parser.add_argument("--window_height").scan<'i', int>().default_value(1080);
    parser.add_argument("--resources_folder").required();

    // Model options
    parser.add_argument("--hp", "--hyper_parameters")
        .action(parse_key_value)
        .append()
        .default_value<hyper_params_vector>({});
    parser.add_argument("--state_dict_folder").required();
    parser.add_argument("--cuda").implicit_value(true).default_value(false);

    parser.parse_args(argc, argv);

    std::map<std::string, std::string> hyper_params;
    for (const auto &[key, value]: parser.get<hyper_params_vector>("--hyper_parameters"))
        hyper_params[key] = value;

    game_loop(
        {parser.get<float>("--wanted_frequency"), parser.get<int>("--nb_tanks"),
         parser.get<int>("--window_width"), parser.get<int>("--window_height"),
         std::filesystem::path(parser.get<std::string>("--resources_folder"))},
        {parser.get<int>("--vision_height"), parser.get<int>("--vision_width"), hyper_params,
         std::filesystem::path(parser.get<std::string>("--state_dict_folder")),
         parser.get<bool>("--cuda")});

    return 0;
}
