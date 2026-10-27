//
// Created by claude on 27/10/2026.
//

#include "./liquid_impala_ppo_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    LiquidImpalaPpoTorchAgentFactory::LiquidImpalaPpoTorchAgentFactory(
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions, const torch::Device device,
        const LiquidImpalaPpoHyperParams &params)
        : config(cli_fields_to_json(liquid_impala_ppo_cli_fields(), params)),
          actor(std::make_shared<LiquidImpalaActor>(
              nb_sensors, nb_continuous_actions, nb_discrete_actions, params.hidden_size_sensors,
              params.hidden_size, params.vision_channels, params.neuron_number,
              params.unfolding_steps, params.delta_t, params.initial_sigma,
              std::vector{params.initial_fire_proba, params.initial_zoom_proba})),
          hidden_state(std::make_shared<LiquidImpalaHiddenState>(actor)),
          rollout_buffer(std::make_shared<LiquidPpoRolloutBuffer>()),
          collector(std::make_shared<LiquidImpalaPpoStepCollector>(rollout_buffer, hidden_state)),
          agent(
              std::make_shared<TorchLiquidImpalaPpoAgent>(actor, hidden_state, device, collector)),
          trainer(std::make_shared<LiquidImpalaPpoTrainer>(
              actor, rollout_buffer, nb_sensors, params.actor_learning_rate,
              params.critic_learning_rate, params.hidden_size_sensors, params.hidden_size,
              params.vision_channels, params.neuron_number, params.unfolding_steps, params.delta_t,
              device, params.metric_window_size, params.gamma, params.gae_lambda,
              params.clip_epsilon, params.target_kl, params.grad_norm_max,
              params.continuous_entropy_factor, params.discrete_entropy_factor, params.epochs,
              params.rollout_size, params.minibatch_size, params.chunk_size)) {}

    std::shared_ptr<AbstractTorchAgent> LiquidImpalaPpoTorchAgentFactory::get_agent() {
        return agent;
    }

    std::shared_ptr<AbstractStepCollector> LiquidImpalaPpoTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> LiquidImpalaPpoTorchAgentFactory::get_trainer() {
        return trainer;
    }

    nlohmann::json LiquidImpalaPpoTorchAgentFactory::get_config() const { return config; }

}// namespace arenai::agent
