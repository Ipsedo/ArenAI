//
// Created by claude on 27/10/2026.
//

#ifndef ARENAI_LIQUID_IMPALA_PPO_FACTORY_H
#define ARENAI_LIQUID_IMPALA_PPO_FACTORY_H

#include "../ppo_liquid/liquid_ppo_rollout_buffer.h"
#include "../torch_factory.h"
#include "./liquid_impala_hidden_state.h"
#include "./liquid_impala_ppo_agent.h"
#include "./liquid_impala_ppo_collector.h"
#include "./liquid_impala_ppo_hyperparams.h"
#include "./liquid_impala_ppo_trainer.h"

namespace arenai::agent {

    class LiquidImpalaPpoTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        LiquidImpalaPpoTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, torch::Device device,
            const LiquidImpalaPpoHyperParams &params);

        std::shared_ptr<AbstractTorchAgent> get_agent() override;
        std::shared_ptr<AbstractStepCollector> get_collector() override;
        std::shared_ptr<AbstractTrainer> get_trainer() override;

        nlohmann::json get_config() const override;

    private:
        nlohmann::json config;

        // triad built once, sharing actor + hidden state + rollout_buffer
        std::shared_ptr<LiquidImpalaActor> actor;
        std::shared_ptr<LiquidImpalaHiddenState> hidden_state;

        std::shared_ptr<LiquidPpoRolloutBuffer> rollout_buffer;
        std::shared_ptr<LiquidImpalaPpoStepCollector> collector;
        std::shared_ptr<TorchLiquidImpalaPpoAgent> agent;
        std::shared_ptr<LiquidImpalaPpoTrainer> trainer;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_IMPALA_PPO_FACTORY_H
