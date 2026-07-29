//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_FACTORY_H
#define ARENAI_PPO_FACTORY_H

#include "../torch_factory.h"
#include "./ppo_agent.h"
#include "./ppo_collector.h"
#include "./ppo_hyperparams.h"
#include "./ppo_rollout_buffer.h"
#include "./ppo_trainer.h"

namespace arenai::agent {

    class PpoTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        PpoTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, torch::Device device, const PpoHyperParams &params);

        std::shared_ptr<AbstractTorchAgent> get_agent() override;
        std::shared_ptr<AbstractStepCollector> get_collector() override;
        std::shared_ptr<AbstractTrainer> get_trainer() override;

    private:
        // triad built once, sharing actor + rollout_buffer
        std::shared_ptr<Actor> actor;

        std::shared_ptr<PpoRolloutBuffer> rollout_buffer;
        std::shared_ptr<PpoStepCollector> collector;
        std::shared_ptr<TorchPpoAgent> agent;
        std::shared_ptr<PpoTrainer> trainer;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_FACTORY_H
