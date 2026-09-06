//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_PPO_AGENT_H
#define ARENAI_LIQUID_PPO_AGENT_H

#include <arenai_agent/agent.h>

#include "../../networks/recurrent/liquid_actor.h"
#include "../torch_agent.h"
#include "./liquid_hidden_state.h"
#include "./liquid_ppo_collector.h"

namespace arenai::agent {

    class TorchLiquidPpoAgent final : public virtual AbstractAgent,
                                      public virtual AbstractTorchAgent {
    public:
        TorchLiquidPpoAgent(
            const std::shared_ptr<LiquidActor> &actor,
            const std::shared_ptr<LiquidHiddenState> &hidden_state, torch::Device device,
            std::optional<std::shared_ptr<LiquidPpoStepCollector>> collector = std::nullopt);

        TorchAction act(const TorchState &state, bool sample) override;

        std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) override;
        void load(const std::filesystem::path &agent_folder) override;

    private:
        std::shared_ptr<LiquidActor> actor;
        std::shared_ptr<LiquidHiddenState> hidden_state;
        std::optional<std::shared_ptr<LiquidPpoStepCollector>> collector;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_AGENT_H
