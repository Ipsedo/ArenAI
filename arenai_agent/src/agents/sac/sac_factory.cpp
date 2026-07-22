//
// Created by claude on 22/07/2026.
//

#include "./sac_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    SacTorchAgentFactory::SacTorchAgentFactory(
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions, const torch::Device device,
        const SacHyperParams &params)
        : actor(std::make_shared<Actor>(
            vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
            params.hidden_size_sensors, params.actor_hidden_sizes, params.vision_channels,
            params.group_norm_nums)),
          replay_buffer(std::make_shared<SacReplayBuffer>(params.replay_buffer_size)),
          collector(std::make_shared<SacStepCollector>(replay_buffer)),
          agent(std::make_shared<TorchSacAgent>(actor, device, collector)),
          trainer(std::make_shared<SacTrainer>(
              actor, replay_buffer, vision_height, vision_width, nb_sensors, nb_continuous_actions,
              nb_discrete_actions, params.actor_learning_rate, params.critic_learning_rate,
              params.alpha_learning_rate, params.hidden_size_sensors, params.hidden_size_actions,
              params.critic_hidden_sizes, params.vision_channels, params.group_norm_nums, device,
              params.metric_window_size, params.tau, params.gamma, params.train_every,
              params.epochs, params.batch_size)) {}

    std::shared_ptr<AbstractTorchAgent> SacTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> SacTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> SacTorchAgentFactory::get_trainer() { return trainer; }

}// namespace arenai::agent
