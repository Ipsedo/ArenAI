#include "./sac_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    SacTorchAgentFactory::SacTorchAgentFactory(
        const int nb_sensors, const int nb_continuous_actions, const int nb_discrete_actions,
        const torch::Device device, const SacHyperParams &params)
        : config(cli_fields_to_json(sac_cli_fields(), params)),
          panorama_encoder(std::make_shared<PanoramaMaskNetwork>(
              params.panorama_channels, params.panorama_group_norm_nums,
              params.panorama_output_channels)),
          panorama_mask_learner(std::make_shared<PanoramaMaskLearner>(
              panorama_encoder, params.panorama_mask_learning_rate, params.panorama_mask_batch_size,
              params.panorama_mask_epochs, device)),
          encoder(std::make_shared<StateEncoder>(
              nb_sensors, params.hidden_size_sensors, params.vision_channels,
              params.group_norm_nums, panorama_encoder)),
          actor(std::make_shared<SacActor>(
              encoder, nb_continuous_actions, nb_discrete_actions, params.actor_hidden_sizes,
              params.initial_sigma,
              std::vector{params.initial_fire_proba, params.initial_zoom_proba})),
          replay_buffer(std::make_shared<SacReplayBuffer>(
              params.buffer_size, params.armed_fraction, params.n_step, params.gamma)),
          collector(std::make_shared<SacStepCollector>(replay_buffer)),
          agent(std::make_shared<TorchSacAgent>(actor, device, collector)),
          trainer(std::make_shared<SacTrainer>(
              actor, encoder, panorama_encoder, panorama_mask_learner, replay_buffer, nb_sensors,
              nb_continuous_actions, nb_discrete_actions, params.actor_learning_rate,
              params.critic_learning_rate, params.alpha_learning_rate, params.hidden_size_sensors,
              params.critic_hidden_sizes, params.vision_channels, params.group_norm_nums, device,
              params.metric_window_size, params.tau, params.initial_alpha, params.target_sigma,
              params.discrete_target_entropy_ratio, params.learning_starts, params.gradient_steps,
              params.batch_size, params.gpu_batch_size)) {}

    std::shared_ptr<AbstractTorchAgent> SacTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> SacTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> SacTorchAgentFactory::get_trainer() { return trainer; }

    std::shared_ptr<PanoramaMaskLearner> SacTorchAgentFactory::get_panorama_mask_learner() {
        return panorama_mask_learner;
    }

    nlohmann::json SacTorchAgentFactory::get_config() const { return config; }

}// namespace arenai::agent
