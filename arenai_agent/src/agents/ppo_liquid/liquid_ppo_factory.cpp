#include "./liquid_ppo_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    LiquidPpoTorchAgentFactory::LiquidPpoTorchAgentFactory(
        const int nb_sensors, const int nb_continuous_actions, const int nb_discrete_actions,
        const torch::Device device, const LiquidPpoHyperParams &params)
        : config(cli_fields_to_json(liquid_ppo_cli_fields(), params)),
          panorama_encoder(std::make_shared<PanoramaMaskNetwork>(
              params.panorama_channels, params.panorama_group_norm_nums,
              params.panorama_output_channels)),
          panorama_mask_learner(std::make_shared<PanoramaMaskLearner>(
              panorama_encoder, params.panorama_mask_learning_rate, params.panorama_mask_batch_size,
              params.panorama_mask_epochs, device)),
          actor(std::make_shared<LiquidActor>(
              nb_sensors, nb_continuous_actions, nb_discrete_actions, params.hidden_size_sensors,
              params.vision_channels, params.group_norm_nums, panorama_encoder,
              params.neuron_number, params.unfolding_steps, params.delta_t, params.initial_sigma,
              std::vector{params.initial_fire_proba, params.initial_zoom_proba})),
          hidden_state(std::make_shared<LiquidHiddenState>(actor)),
          rollout_buffer(std::make_shared<LiquidPpoRolloutBuffer>()),
          collector(std::make_shared<LiquidPpoStepCollector>(rollout_buffer, hidden_state)),
          agent(std::make_shared<TorchLiquidPpoAgent>(actor, hidden_state, device, collector)),
          trainer(std::make_shared<LiquidPpoTrainer>(
              actor, panorama_encoder, panorama_mask_learner, rollout_buffer, nb_continuous_actions,
              nb_discrete_actions, nb_sensors, params.actor_learning_rate,
              params.critic_learning_rate, params.hidden_size_sensors, params.vision_channels,
              params.group_norm_nums, params.neuron_number, params.unfolding_steps, params.delta_t,
              device, params.metric_window_size, params.gamma, params.gae_lambda,
              params.clip_epsilon, params.target_kl, params.grad_norm_max, params.initial_alpha,
              params.alpha_learning_rate, params.continuous_target_entropy,
              params.discrete_target_proba, params.actor_epochs, params.critic_epochs,
              params.rollout_size, params.actor_batch_size, params.actor_gpu_batch_size,
              params.critic_batch_size, params.critic_gpu_batch_size, params.chunk_size)) {}

    std::shared_ptr<AbstractTorchAgent> LiquidPpoTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> LiquidPpoTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> LiquidPpoTorchAgentFactory::get_trainer() { return trainer; }

    std::shared_ptr<PanoramaMaskLearner> LiquidPpoTorchAgentFactory::get_panorama_mask_learner() {
        return panorama_mask_learner;
    }

    nlohmann::json LiquidPpoTorchAgentFactory::get_config() const { return config; }

}// namespace arenai::agent
