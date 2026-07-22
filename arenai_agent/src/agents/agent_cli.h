//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_ALGORITHMS_H
#define ARENAI_ALGORITHMS_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <argparse/argparse.hpp>
#include <torch/torch.h>

#include "./torch_factory.h"

namespace arenai::agent {

    inline constexpr std::string DEFAULT_AGENT = "sac";

    // One RL algorithm exposed as a CLI subcommand: its subparser (owning the
    // algorithm's hyper-parameter options) and the builder reading them back
    // to construct the concrete factory.
    struct AgentCli {
        std::string name;
        // stable address: the main parser keeps a reference on the subparser
        std::unique_ptr<argparse::ArgumentParser> parser;
        std::function<std::unique_ptr<AbstractTorchAgentFactory>(
            int vision_height, int vision_width, torch::Device device)>
            create_factory;
    };

    std::vector<AgentCli> make_agent_clis();

}// namespace arenai::agent

#endif//ARENAI_ALGORITHMS_H
