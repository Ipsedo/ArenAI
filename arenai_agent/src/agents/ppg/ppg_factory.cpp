//
// Created by samuel on 22/07/2026.
//

#include "./ppg_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    PpgTorchAgentFactory::PpgTorchAgentFactory(
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions, const torch::Device device,
        const PpgHyperParams &params)
        : config(cli_fields_to_json(ppg_cli_fields(), params)),
          actor(std::make_shared<DistillationActor>(
              nb_sensors, nb_continuous_actions, nb_discrete_actions, params.hidden_size_sensors,
              params.actor_hidden_sizes, params.vision_channels, params.group_norm_nums,
              params.initial_sigma,
              std::vector{params.initial_fire_proba, params.initial_zoom_proba})),
          rollout_buffer(std::make_shared<PpgRolloutBuffer>()),
          collector(std::make_shared<PpgStepCollector>(rollout_buffer)),
          agent(std::make_shared<TorchPpgAgent>(actor, device, collector)),
          trainer(std::make_shared<PpgTrainer>(
              actor, rollout_buffer, nb_sensors, params.actor_learning_rate,
              params.critic_learning_rate, params.hidden_size_sensors, params.critic_hidden_sizes,
              params.vision_channels, params.group_norm_nums, device, params.metric_window_size,
              params.gamma, params.gae_lambda, params.clip_epsilon, params.target_kl,
              params.grad_norm_max, params.continuous_entropy_factor,
              params.discrete_entropy_factor, params.actor_epochs, params.critic_epochs,
              params.rollout_size, params.actor_batch_size, params.actor_gpu_batch_size,
              params.critic_batch_size, params.critic_gpu_batch_size, params.actor_aux_epochs,
              params.critic_aux_epochs, params.beta_clone)) {}

    std::shared_ptr<AbstractTorchAgent> PpgTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> PpgTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> PpgTorchAgentFactory::get_trainer() { return trainer; }

    nlohmann::json PpgTorchAgentFactory::get_config() const { return config; }

}// namespace arenai::agent
