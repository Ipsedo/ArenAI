//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_SAC_FACTORY_H
#define ARENAI_SAC_FACTORY_H

#include "../torch_factory.h"
#include "./replay_buffer.h"
#include "./sac.h"
#include "./sac_collector.h"
#include "./sac_trainer.h"

namespace arenai::train {

    class SacTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        SacTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, float actor_learning_rate, float critic_learning_rate,
            float alpha_learning_rate, int hidden_size_sensors, int hidden_size_actions,
            const std::vector<int> &actor_hidden_sizes, const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float tau, float gamma, int replay_buffer_size, int train_every, int epochs,
            int batch_size);

        std::shared_ptr<AbstractTorchAgent> get_agent() override;
        std::shared_ptr<AbstractStepCollector> get_collector() override;
        std::shared_ptr<AbstractTrainer> get_trainer() override;

    private:
        // triad built once, sharing actor + replay_buffer
        std::shared_ptr<Actor> actor;

        std::shared_ptr<SacReplayBuffer> replay_buffer;
        std::shared_ptr<SacStepCollector> collector;
        std::shared_ptr<TorchSacAgent> agent;
        std::shared_ptr<SacTrainer> trainer;
    };

}// namespace arenai::train

#endif//ARENAI_SAC_FACTORY_H
