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

    struct AgentCli {
        std::string name;
        std::unique_ptr<argparse::ArgumentParser> parser;
        std::function<std::unique_ptr<AbstractTorchAgentFactory>(
            int vision_height, int vision_width, torch::Device device)>
            create_factory;
    };

    std::vector<AgentCli> make_agent_clis();

}// namespace arenai::agent

#endif//ARENAI_ALGORITHMS_H
