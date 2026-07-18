//
// Created by samuel on 11/03/2026.
//

#include <arenai_train/factory_set.h>

#include "../utils/cli_parser.h"
#include "./sac.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    SacAgentFactory::SacAgentFactory(const std::map<std::string, std::string> &arguments)
        : AgentFactory(arguments) {}

    std::shared_ptr<AbstractAgent> SacAgentFactory::get_agent_impl(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_continuous_actions, const int &nb_discrete_action) {
        return std::make_shared<SacAgent>(
            vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_action,
            get_value("actor_learning_rate", 1e-4f), get_value("critic_learning_rate", 3e-4f),
            get_value("alpha_learning_rate", 3e-4f), get_value("sensors_hidden_size", 256),
            get_value("actions_hidden_size", 64),
            get_value<hidden_layers>("actor_hidden_size", parse_cli_hidden_layer, {{2560, 1024}})
                .layers,
            get_value<hidden_layers>("critic_hidden_size", parse_cli_hidden_layer, {{2560, 1024}})
                .layers,
            get_value<vision_channels>(
                "vision_channels", parse_cli_vision_channels,
                {{{3, 8}, {8, 16}, {16, 32}, {32, 64}, {64, 128}, {128, 256}}})
                .channels,
            get_value<group_norm_nums>(
                "group_norm_nums", parse_cli_group_norms, {{{1, 2, 4, 8, 16, 32}}})
                .groups,
            torch::Device(torch::kCPU), get_value("metric_window_size", 1024),
            get_value("tau", 0.005f), get_value("gamma", 0.995f));
    }

}// namespace arenai::train
