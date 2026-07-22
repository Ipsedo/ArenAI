//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_FACTORY_H
#define ARENAI_PPO_FACTORY_H

#include "../torch_factory.h"
#include "./ppo_agent.h"
#include "./ppo_collector.h"
#include "./ppo_rollout_buffer.h"
#include "./ppo_trainer.h"

namespace arenai::agent {

    class PpoTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        PpoTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, float actor_learning_rate, float critic_learning_rate,
            int hidden_size_sensors, int hidden_size_actions,
            const std::vector<int> &actor_hidden_sizes, const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float gamma, float gae_lambda, float clip_epsilon, float entropy_coef, int epochs,
            int batch_size);

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
