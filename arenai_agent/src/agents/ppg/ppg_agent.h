//
// Created by samuel on 22/07/2026.
//

#ifndef ARENAI_PPG_AGENT_H
#define ARENAI_PPG_AGENT_H

#include <arenai_agent/agent.h>

#include "../../networks/actor.h"
#include "../torch_agent.h"
#include "./ppg_collector.h"

namespace arenai::agent {

    class TorchPpgAgent final : public virtual AbstractAgent, public virtual AbstractTorchAgent {
    public:
        TorchPpgAgent(
            const std::shared_ptr<DistillationActor> &actor, torch::Device device,
            std::optional<std::shared_ptr<PpgStepCollector>> collector = std::nullopt);

        TorchAction act(const TorchState &state, bool sample) override;

        std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) override;
        void load(const std::filesystem::path &agent_folder) override;

    private:
        std::shared_ptr<DistillationActor> actor;
        std::optional<std::shared_ptr<PpgStepCollector>> collector;
    };

}// namespace arenai::agent

#endif//ARENAI_PPG_AGENT_H
