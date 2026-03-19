//
// Created by samuel on 11/03/2026.
//

#include "./cli_parser.h"

#include <regex>

vision_channels parse_cli_vision_channels(const std::string &value) {
    const std::regex regex_match(
        R"(^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$)");
    const std::regex regex_layer(R"(\( *\d+ *, *\d+ *\))");
    const std::regex regex_channel(R"(\d+)");

    if (!std::regex_match(value.begin(), value.end(), regex_match))
        throw std::invalid_argument(
            "invalid --vision_channels format, usage : [(10, 20), (20, 40), ...], actual value = \""
            + value + "\"");

    vision_channels vision_channels;

    std::sregex_iterator it_layer(value.begin(), value.end(), regex_layer);
    for (const std::sregex_iterator end; it_layer != end; ++it_layer) {
        const auto layer_str = it_layer->str();

        const std::sregex_iterator it_channel(layer_str.begin(), layer_str.end(), regex_channel);

        const int c_i = std::stoi(it_channel->str());
        const int c_o = std::stoi(std::next(it_channel)->str());

        vision_channels.channels.emplace_back(c_i, c_o);
    }

    return vision_channels;
}

group_norm_nums parse_cli_group_norms(const std::string &value) {
    const std::regex regex_match(R"(^ *\[(?: *\d+ *,)* *\d+ *] *$)");
    const std::regex regex_groups(R"(\d+)");

    if (!std::regex_match(value.begin(), value.end(), regex_match))
        throw std::invalid_argument(
            "invalid --group_norm_nums format, usage : [4, 8, 16, ...], actual value = \"" + value
            + "\"");

    group_norm_nums group_nums;

    std::sregex_iterator it_layer(value.begin(), value.end(), regex_groups);
    for (const std::sregex_iterator end; it_layer != end; ++it_layer)
        group_nums.groups.emplace_back(std::stoi(it_layer->str()));

    return group_nums;
}
