//
// Created by claude on 26/10/2026.
//

#include "./impala_ppo_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    ImpalaPpoTorchAgentFactory::ImpalaPpoTorchAgentFactory(
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions, const torch::Device device,
        const ImpalaPpoHyperParams &params)
        : config(cli_fields_to_json(impala_ppo_cli_fields(), params)),
          actor(std::make_shared<ImpalaActor>(
              nb_sensors, nb_continuous_actions, nb_discrete_actions, params.hidden_size_sensors,
              params.actor_hidden_sizes, params.vision_channels, params.initial_sigma,
              std::vector{params.initial_fire_proba, params.initial_zoom_proba})),
          rollout_buffer(std::make_shared<PpoRolloutBuffer>()),
          collector(std::make_shared<PpoStepCollector>(rollout_buffer)),
          agent(std::make_shared<TorchImpalaPpoAgent>(actor, device, collector)),
          trainer(std::make_shared<ImpalaPpoTrainer>(
              actor, rollout_buffer, nb_sensors, params.actor_learning_rate,
              params.critic_learning_rate, params.hidden_size_sensors, params.critic_hidden_sizes,
              params.vision_channels, device, params.metric_window_size, params.gamma,
              params.gae_lambda, params.clip_epsilon, params.target_kl, params.grad_norm_max,
              params.continuous_entropy_factor, params.discrete_entropy_factor, params.actor_epochs,
              params.critic_epochs, params.rollout_size, params.minibatch_size)) {}

    std::shared_ptr<AbstractTorchAgent> ImpalaPpoTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> ImpalaPpoTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> ImpalaPpoTorchAgentFactory::get_trainer() { return trainer; }

    nlohmann::json ImpalaPpoTorchAgentFactory::get_config() const { return config; }

}// namespace arenai::agent
