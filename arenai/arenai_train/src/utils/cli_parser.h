//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_TRAIN_HOST_CLI_PARSER_H
#define ARENAI_TRAIN_HOST_CLI_PARSER_H

#include <string>
#include <vector>

struct vision_channels {
    std::vector<std::tuple<int, int>> channels;
};

struct group_norm_nums {
    std::vector<int> groups;
};

struct hidden_layers {
    std::vector<int> layers;
};

vision_channels parse_cli_vision_channels(const std::string &value);

group_norm_nums parse_cli_group_norms(const std::string &value);
hidden_layers parse_cli_hidden_layer(const std::string &value);

#endif//ARENAI_TRAIN_HOST_CLI_PARSER_H
