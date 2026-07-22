//
// Created by claude on 22/07/2026.
//

#include "./ppo_factory.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    PpoTorchAgentFactory::PpoTorchAgentFactory(
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions,
        const float actor_learning_rate, const float critic_learning_rate,
        const int hidden_size_sensors, const int hidden_size_actions,
        const std::vector<int> &actor_hidden_sizes, const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float gamma, const float gae_lambda,
        const float clip_epsilon, const float continuous_entropy_coef,
        const float discrete_entropy_coef, const int epochs, const int batch_size)
        : actor(std::make_shared<Actor>(
            vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
            hidden_size_sensors, actor_hidden_sizes, vision_channels, group_norm_nums)),
          rollout_buffer(std::make_shared<PpoRolloutBuffer>()),
          collector(std::make_shared<PpoStepCollector>(rollout_buffer)),
          agent(std::make_shared<TorchPpoAgent>(actor, device, collector)),
          trainer(std::make_shared<PpoTrainer>(
              actor, rollout_buffer, vision_height, vision_width, nb_sensors, actor_learning_rate,
              critic_learning_rate, hidden_size_sensors, hidden_size_actions, critic_hidden_sizes,
              vision_channels, group_norm_nums, device, metric_window_size, gamma, gae_lambda,
              clip_epsilon, continuous_entropy_coef, discrete_entropy_coef, epochs, batch_size)) {}

    std::shared_ptr<AbstractTorchAgent> PpoTorchAgentFactory::get_agent() { return agent; }

    std::shared_ptr<AbstractStepCollector> PpoTorchAgentFactory::get_collector() {
        return collector;
    }

    std::shared_ptr<AbstractTrainer> PpoTorchAgentFactory::get_trainer() { return trainer; }

}// namespace arenai::agent
