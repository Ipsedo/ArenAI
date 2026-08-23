//
// Created by claude on 22/07/2026.
//

#include "./ppo_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    PpoTorchAgentFactory::PpoTorchAgentFactory(
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions, const torch::Device device,
        const PpoHyperParams &params)
        : actor(std::make_shared<Actor>(
            vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
            params.hidden_size_sensors, params.actor_hidden_sizes, params.vision_channels,
            params.group_norm_nums, params.initial_sigma, params.initial_fire_proba)),
          rollout_buffer(std::make_shared<PpoRolloutBuffer>()),
          collector(std::make_shared<PpoStepCollector>(rollout_buffer)),
          agent(std::make_shared<TorchPpoAgent>(actor, device, collector)),
          trainer(std::make_shared<PpoTrainer>(
              actor, rollout_buffer, vision_height, vision_width, nb_sensors, nb_continuous_actions,
              params.actor_learning_rate, params.critic_learning_rate, params.hidden_size_sensors,
              params.critic_hidden_sizes, params.vision_channels, params.group_norm_nums, device,
              params.metric_window_size, params.gamma, params.gae_lambda, params.clip_epsilon,
              params.target_kl, params.grad_norm_max, params.target_entropy_init,
              params.target_entropy_final, params.target_entropy_warmup_steps,
              params.target_fire_proba_init, params.target_fire_proba_final,
              params.target_fire_proba_warmup_steps, params.epochs, params.rollout_size,
              params.minibatch_size)) {}

    std::shared_ptr<AbstractTorchAgent> PpoTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> PpoTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> PpoTorchAgentFactory::get_trainer() { return trainer; }

}// namespace arenai::agent
