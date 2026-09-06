//
// Created by samuel on 06/09/2026.
//

#include "./liquid_ppo_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    LiquidPpoTorchAgentFactory::LiquidPpoTorchAgentFactory(
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions, const torch::Device device,
        const LiquidPpoHyperParams &params)
        : config(cli_fields_to_map(liquid_ppo_cli_fields(), params)),
          actor(std::make_shared<LiquidActor>(
              vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
              params.hidden_size_sensors, params.vision_channels, params.group_norm_nums,
              params.neuron_number, params.unfolding_steps, params.delta_t, params.initial_sigma,
              params.initial_fire_proba)),
          hidden_state(std::make_shared<LiquidHiddenState>(actor)),
          rollout_buffer(std::make_shared<LiquidPpoRolloutBuffer>()),
          collector(std::make_shared<LiquidPpoStepCollector>(rollout_buffer, hidden_state)),
          agent(std::make_shared<TorchLiquidPpoAgent>(actor, hidden_state, device, collector)),
          trainer(std::make_shared<LiquidPpoTrainer>(
              actor, rollout_buffer, vision_height, vision_width, nb_sensors, nb_continuous_actions,
              nb_discrete_actions, params.actor_learning_rate, params.critic_learning_rate,
              params.hidden_size_sensors, params.vision_channels, params.group_norm_nums,
              params.neuron_number, params.unfolding_steps, params.delta_t, device,
              params.metric_window_size, params.gamma, params.gae_lambda, params.clip_epsilon,
              params.target_kl, params.grad_norm_max, params.continuous_target_entropy,
              params.discrete_target_entropy_factor, params.epochs, params.rollout_size,
              params.minibatch_size, params.chunk_size)) {}

    std::shared_ptr<AbstractTorchAgent> LiquidPpoTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> LiquidPpoTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> LiquidPpoTorchAgentFactory::get_trainer() { return trainer; }

    std::map<std::string, std::string> LiquidPpoTorchAgentFactory::get_config() const {
        return config;
    }

}// namespace arenai::agent
