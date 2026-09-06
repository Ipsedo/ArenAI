//
// Created by samuel on 22/01/2026.
//

#include <format>
#include <iostream>
#include <map>
#include <string>

#include <arenai_agent/factory.h>

#include "../utils/cli_parser.h"
#include "./ppo/ppo_agent.h"
#include "./ppo_liquid/liquid_ppo_agent.h"
#include "./sac/sac_agent.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    AgentFactory::AgentFactory(const std::map<std::string, std::string> &arguments)
        : arguments(arguments) {}

    std::shared_ptr<AbstractAgent> AgentFactory::get_agent(
        const AgentAlgorithm algorithm, const int &vision_height, const int &vision_width,
        const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_actions) {
        std::shared_ptr<AbstractAgent> agent;
        switch (algorithm) {
            case SAC:
                agent = create_sac_agent(
                    vision_height, vision_width, nb_sensors, nb_continuous_actions,
                    nb_discrete_actions);
                break;
            case PPO:
                agent = create_ppo_agent(
                    vision_height, vision_width, nb_sensors, nb_continuous_actions,
                    nb_discrete_actions);
                break;
            case PPO_LIQUID:
                agent = create_liquid_ppo_agent(
                    vision_height, vision_width, nb_sensors, nb_continuous_actions,
                    nb_discrete_actions);
                break;
            default: throw std::runtime_error("Unknown agent algorithm");
        }

        if (!arguments.empty()) {
            std::cerr << "Invalid argument(s) : " << std::get<0>(*arguments.begin()) << std::endl;
            throw std::runtime_error("Invalid argument(s)");
        }

        return agent;
    }

    std::shared_ptr<AbstractAgent> AgentFactory::create_sac_agent(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_continuous_actions, const int &nb_discrete_action) {
        return std::make_shared<TorchSacAgent>(
            std::make_shared<Actor>(
                vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_action,
                get_value("hidden_size_sensors", 128),
                get_value<hidden_layers>("hidden_sizes", parse_cli_hidden_layer, {{1024, 512}})
                    .layers,
                get_value<vision_channels>(
                    "vision_channels", parse_cli_vision_channels,
                    {{{3, 8}, {8, 16}, {16, 24}, {24, 32}, {32, 48}, {48, 64}}})
                    .channels,
                get_value<group_norm_nums>(
                    "group_norm_nums", parse_cli_group_norms, {{{1, 2, 3, 4, 6, 8}}})
                    .groups,
                0.f, 0.f),
            get_value<bool>("cuda", false) ? torch::kCUDA : torch::kCPU);
    }

    std::shared_ptr<AbstractAgent> AgentFactory::create_ppo_agent(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_continuous_actions, const int &nb_discrete_action) {
        return std::make_shared<TorchPpoAgent>(
            std::make_shared<Actor>(
                vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_action,
                get_value("hidden_size_sensors", 128),
                get_value<hidden_layers>("hidden_sizes", parse_cli_hidden_layer, {{1024, 512}})
                    .layers,
                get_value<vision_channels>(
                    "vision_channels", parse_cli_vision_channels,
                    {{{3, 8}, {8, 16}, {16, 24}, {24, 32}, {32, 48}, {48, 64}}})
                    .channels,
                get_value<group_norm_nums>(
                    "group_norm_nums", parse_cli_group_norms, {{{1, 2, 3, 4, 6, 8}}})
                    .groups,
                0.f, 0.f),
            get_value<bool>("cuda", false) ? torch::kCUDA : torch::kCPU);
    }

    std::shared_ptr<AbstractAgent> AgentFactory::create_liquid_ppo_agent(
        const int &vision_height, const int &vision_width, const int &nb_sensors,
        const int &nb_continuous_actions, const int &nb_discrete_action) {
        const auto actor = std::make_shared<LiquidActor>(
            vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_action,
            get_value("hidden_size_sensors", 128),
            get_value<vision_channels>(
                "vision_channels", parse_cli_vision_channels,
                {{{3, 8}, {8, 16}, {16, 24}, {24, 32}, {32, 48}, {48, 64}}})
                .channels,
            get_value<group_norm_nums>(
                "group_norm_nums", parse_cli_group_norms, {{{1, 2, 3, 4, 6, 8}}})
                .groups,
            get_value("neuron_number", 128), get_value("unfolding_steps", 6),
            get_value("delta_t", 1.f / 30.f), 0.f, 0.f);

        return std::make_shared<TorchLiquidPpoAgent>(
            actor, std::make_shared<LiquidHiddenState>(actor),
            get_value<bool>("cuda", false) ? torch::kCUDA : torch::kCPU);
    }

}// namespace arenai::agent
