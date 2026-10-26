//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_IMPALA_PPO_FACTORY_H
#define ARENAI_IMPALA_PPO_FACTORY_H

#include "../ppo/ppo_collector.h"
#include "../ppo/ppo_rollout_buffer.h"
#include "../torch_factory.h"
#include "./impala_ppo_agent.h"
#include "./impala_ppo_hyperparams.h"
#include "./impala_ppo_trainer.h"

namespace arenai::agent {

    class ImpalaPpoTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        ImpalaPpoTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, torch::Device device, const ImpalaPpoHyperParams &params);

        std::shared_ptr<AbstractTorchAgent> get_agent() override;
        std::shared_ptr<AbstractStepCollector> get_collector() override;
        std::shared_ptr<AbstractTrainer> get_trainer() override;

        nlohmann::json get_config() const override;

    private:
        nlohmann::json config;

        // triad built once, sharing actor + rollout_buffer
        std::shared_ptr<ImpalaActor> actor;

        std::shared_ptr<PpoRolloutBuffer> rollout_buffer;
        std::shared_ptr<PpoStepCollector> collector;
        std::shared_ptr<TorchImpalaPpoAgent> agent;
        std::shared_ptr<ImpalaPpoTrainer> trainer;
    };

}// namespace arenai::agent

#endif//ARENAI_IMPALA_PPO_FACTORY_H
