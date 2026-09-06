//
// Created by samuel on 22/01/2026.
//

#include <string>

#include <arenai_agent/factory.h>

#include "../utils/cli_parser.h"
#include "./ppo/ppo_agent.h"
#include "./ppo_liquid/liquid_ppo_agent.h"
#include "./sac/sac_agent.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    AgentFactory::AgentFactory(const nlohmann::json &config)
        : agent_arguments(config.at("agent")),
          vision_height(config.at("environment").at("vision_height").get<int>()),
          vision_width(config.at("environment").at("vision_width").get<int>()),
          wanted_frequency(config.at("environment").at("wanted_frequency").get<float>()) {}

    int AgentFactory::get_vision_height() const { return vision_height; }
    int AgentFactory::get_vision_width() const { return vision_width; }
    float AgentFactory::get_wanted_frequency() const { return wanted_frequency; }

    std::shared_ptr<AbstractAgent> AgentFactory::get_agent(
        const AgentAlgorithm algorithm, const int &nb_sensors, const int &nb_continuous_actions,
        const int &nb_discrete_actions, const bool cuda) {
        switch (algorithm) {
            case SAC:
                return create_sac_agent(
                    nb_sensors, nb_continuous_actions, nb_discrete_actions, cuda);
            case PPO:
                return create_ppo_agent(
                    nb_sensors, nb_continuous_actions, nb_discrete_actions, cuda);
            case PPO_LIQUID:
                return create_liquid_ppo_agent(
                    nb_sensors, nb_continuous_actions, nb_discrete_actions, cuda);
            default: throw std::runtime_error("Unknown agent algorithm");
        }
    }

    std::shared_ptr<AbstractAgent> AgentFactory::create_sac_agent(
        const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_action,
        const bool cuda) {
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
            cuda ? torch::kCUDA : torch::kCPU);
    }

    std::shared_ptr<AbstractAgent> AgentFactory::create_ppo_agent(
        const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_action,
        const bool cuda) {
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
            cuda ? torch::kCUDA : torch::kCPU);
    }

    std::shared_ptr<AbstractAgent> AgentFactory::create_liquid_ppo_agent(
        const int &nb_sensors, const int &nb_continuous_actions, const int &nb_discrete_action,
        const bool cuda) {
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
            actor, std::make_shared<LiquidHiddenState>(actor), cuda ? torch::kCUDA : torch::kCPU);
    }

}// namespace arenai::agent
