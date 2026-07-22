//
// Created by samuel on 11/03/2026.
//

#include "./cli_parser.h"

#include <regex>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    vision_channels parse_cli_vision_channels(const std::string &value) {
        const std::regex regex_match(
            R"(^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$)");
        const std::regex regex_layer(R"(\( *\d+ *, *\d+ *\))");
        const std::regex regex_channel(R"(\d+)");

        if (!std::regex_match(value.begin(), value.end(), regex_match))
            throw std::invalid_argument(
                "invalid --vision_channels format, usage : [(10, 20), (20, 40), ...], actual value "
                "= \""
                + value + "\"");

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
    }

    std::vector<int> parse_int_vector(
        const std::string &value, const std::string &cli_arg_name,
        const std::string &cli_arg_value_suggestion) {
        const std::regex regex_match(R"(^ *\[(?: *\d+ *,)* *\d+ *] *$)");
        const std::regex regex_groups(R"(\d+)");

        if (!std::regex_match(value.begin(), value.end(), regex_match))
            throw std::invalid_argument(
                "invalid " + cli_arg_name + " format, usage : " + cli_arg_value_suggestion
                + ", actual value = \"" + value + "\"");

        std::vector<int> int_vector;

        std::sregex_iterator it_layer(value.begin(), value.end(), regex_groups);
        for (const std::sregex_iterator end; it_layer != end; ++it_layer)
            int_vector.emplace_back(std::stoi(it_layer->str()));

        return int_vector;
    }

    group_norm_nums parse_cli_group_norms(const std::string &value) {
        return {parse_int_vector(value, "group_norm_nums", "[4, 8, 16, ...]")};
    }

    hidden_layers parse_cli_hidden_layer(const std::string &value) {
        return {parse_int_vector(value, "hidden_layers", "[256, 128, ..., 64]")};
    }

}// namespace arenai::agent
