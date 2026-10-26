//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_IMPALA_PPO_AGENT_H
#define ARENAI_IMPALA_PPO_AGENT_H

#include <arenai_agent/agent.h>

#include "../../networks/impala_actor.h"
#include "../ppo/ppo_collector.h"
#include "../torch_agent.h"

namespace arenai::agent {

    // TorchPpoAgent with an ImpalaActor: the on-policy step collection is the same
    class TorchImpalaPpoAgent final : public virtual AbstractAgent,
                                      public virtual AbstractTorchAgent {
    public:
        TorchImpalaPpoAgent(
            const std::shared_ptr<ImpalaActor> &actor, torch::Device device,
            std::optional<std::shared_ptr<PpoStepCollector>> collector = std::nullopt);

        TorchAction act(const TorchState &state, bool sample) override;

        std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) override;
        void load(const std::filesystem::path &agent_folder) override;

    private:
        std::shared_ptr<ImpalaActor> actor;
        std::optional<std::shared_ptr<PpoStepCollector>> collector;
    };

}// namespace arenai::agent

#endif//ARENAI_IMPALA_PPO_AGENT_H
