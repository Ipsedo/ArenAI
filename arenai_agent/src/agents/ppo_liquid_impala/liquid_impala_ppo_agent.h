//
// Created by claude on 27/10/2026.
//

#ifndef ARENAI_LIQUID_IMPALA_PPO_AGENT_H
#define ARENAI_LIQUID_IMPALA_PPO_AGENT_H

#include <arenai_agent/agent.h>

#include "../../networks/recurrent/liquid_impala_actor.h"
#include "../torch_agent.h"
#include "./liquid_impala_hidden_state.h"
#include "./liquid_impala_ppo_collector.h"

namespace arenai::agent {

    class TorchLiquidImpalaPpoAgent final : public virtual AbstractAgent,
                                            public virtual AbstractTorchAgent {
    public:
        TorchLiquidImpalaPpoAgent(
            const std::shared_ptr<LiquidImpalaActor> &actor,
            const std::shared_ptr<LiquidImpalaHiddenState> &hidden_state, torch::Device device,
            std::optional<std::shared_ptr<LiquidImpalaPpoStepCollector>> collector = std::nullopt);

        TorchAction act(const TorchState &state, bool sample) override;

        std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) override;
        void load(const std::filesystem::path &agent_folder) override;

    private:
        std::shared_ptr<LiquidImpalaActor> actor;
        std::shared_ptr<LiquidImpalaHiddenState> hidden_state;
        std::optional<std::shared_ptr<LiquidImpalaPpoStepCollector>> collector;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_IMPALA_PPO_AGENT_H
