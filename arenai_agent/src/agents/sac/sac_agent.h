#ifndef ARENAI_SAC_AGENT_H
#define ARENAI_SAC_AGENT_H

#include <memory>
#include <optional>

#include <arenai_agent/agent.h>

#include "../../networks/actor.h"
#include "../torch_agent.h"
#include "./sac_collector.h"

namespace arenai::agent {

    class TorchSacAgent final : public virtual AbstractAgent, public virtual AbstractTorchAgent {
    public:
        TorchSacAgent(
            const std::shared_ptr<SacActor> &actor, torch::Device device,
            std::optional<std::shared_ptr<SacStepCollector>> collector = std::nullopt);

        TorchAction act(const TorchState &state, bool sample) override;

        std::vector<core::Action>
        act(const std::vector<core::State> &states, int vision_height, int vision_width) override;
        void load(const std::filesystem::path &agent_folder) override;

    private:
        std::shared_ptr<SacActor> actor;
        std::optional<std::shared_ptr<SacStepCollector>> collector;
    };

}// namespace arenai::agent

#endif//ARENAI_SAC_AGENT_H
