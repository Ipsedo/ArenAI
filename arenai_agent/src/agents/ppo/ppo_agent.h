//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_AGENT_H
#define ARENAI_PPO_AGENT_H

#include <arenai_agent/agent.h>

#include "../../networks/actor.h"
#include "../torch_agent.h"
#include "./ppo_collector.h"

namespace arenai::agent {

    class TorchPpoAgent final : public virtual AbstractAgent, public virtual AbstractTorchAgent {
    public:
        TorchPpoAgent(
            const std::shared_ptr<Actor> &actor, torch::Device device,
            std::optional<std::shared_ptr<PpoStepCollector>> collector = std::nullopt);

        TorchAction act(const TorchState &state) override;

        std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) override;
        void load(const std::filesystem::path &agent_folder) override;

    private:
        std::shared_ptr<Actor> actor;
        std::optional<std::shared_ptr<PpoStepCollector>> collector;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_AGENT_H
