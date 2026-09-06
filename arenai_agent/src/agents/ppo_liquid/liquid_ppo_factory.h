//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_PPO_FACTORY_H
#define ARENAI_LIQUID_PPO_FACTORY_H

#include "../torch_factory.h"
#include "./liquid_hidden_state.h"
#include "./liquid_ppo_agent.h"
#include "./liquid_ppo_collector.h"
#include "./liquid_ppo_hyperparams.h"
#include "./liquid_ppo_rollout_buffer.h"
#include "./liquid_ppo_trainer.h"

namespace arenai::agent {

    class LiquidPpoTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        LiquidPpoTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, torch::Device device, const LiquidPpoHyperParams &params);

        std::shared_ptr<AbstractTorchAgent> get_agent() override;
        std::shared_ptr<AbstractStepCollector> get_collector() override;
        std::shared_ptr<AbstractTrainer> get_trainer() override;

        std::map<std::string, std::string> get_config() const override;

    private:
        std::map<std::string, std::string> config;

        // triad built once, sharing actor + hidden state + rollout_buffer
        std::shared_ptr<LiquidActor> actor;
        std::shared_ptr<LiquidHiddenState> hidden_state;

        std::shared_ptr<LiquidPpoRolloutBuffer> rollout_buffer;
        std::shared_ptr<LiquidPpoStepCollector> collector;
        std::shared_ptr<TorchLiquidPpoAgent> agent;
        std::shared_ptr<LiquidPpoTrainer> trainer;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_FACTORY_H
