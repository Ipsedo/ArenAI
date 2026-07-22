//
// Created by claude on 22/07/2026.
//

#include "./agent_cli.h"

#include <arenai_core/constants.h>

#include "../utils/cli_fields.h"
#include "./ppo/ppo_factory.h"
#include "./ppo/ppo_hyperparams.h"
#include "./sac/sac_factory.h"
#include "./sac/sac_hyperparams.h"

namespace arenai::agent {

    namespace {

        template<typename HyperParams, typename Factory>
        AgentCli
        make_agent_cli(const std::string &name, std::vector<CliField<HyperParams>> fields) {
            auto parser = std::make_unique<argparse::ArgumentParser>(name);
            add_cli_fields(*parser, fields);

            auto *parser_ptr = parser.get();
            return {
                name, std::move(parser),
                [parser_ptr, fields = std::move(fields)](
                    const int vision_height, const int vision_width, const torch::Device device) {
                    return std::make_unique<Factory>(
                        vision_height, vision_width, model::ENEMY_PROPRIOCEPTION_SIZE,
                        model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION, device,
                        read_cli_fields(*parser_ptr, fields));
                }};
        }

    }// namespace

    std::vector<AgentCli> make_agent_clis() {
        std::vector<AgentCli> algorithms;

        algorithms.push_back(
            make_agent_cli<SacHyperParams, SacTorchAgentFactory>("sac", sac_cli_fields()));
        algorithms.push_back(
            make_agent_cli<PpoHyperParams, PpoTorchAgentFactory>("ppo", ppo_cli_fields()));

        return algorithms;
    }

}// namespace arenai::agent
