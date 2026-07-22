//
// Created by samuel on 21/01/2026.
//

#ifndef ARENAI_AGENT_HOST_SAC_H
#define ARENAI_AGENT_HOST_SAC_H

#include <arenai_agent/agent.h>

#include "../../networks/actor.h"
#include "../torch_agent.h"
#include "./sac_collector.h"

namespace arenai::agent {

    class TorchSacAgent final : public virtual AbstractAgent, public virtual AbstractTorchAgent {
    public:
        TorchSacAgent(
            const std::shared_ptr<Actor> &actor, torch::Device device,
            std::optional<std::shared_ptr<SacStepCollector>> collector = std::nullopt);

        TorchAction act(const TorchState &state) override;

        std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) override;
        void load(const std::filesystem::path &agent_folder) override;

    private:
        std::shared_ptr<Actor> actor;
        std::optional<std::shared_ptr<SacStepCollector>> collector;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_SAC_H
