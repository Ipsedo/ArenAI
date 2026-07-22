//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_SAC_FACTORY_H
#define ARENAI_SAC_FACTORY_H

#include "../torch_factory.h"
#include "./sac_agent.h"
#include "./sac_collector.h"
#include "./sac_hyperparams.h"
#include "./sac_replay_buffer.h"
#include "./sac_trainer.h"

namespace arenai::agent {

    class SacTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        SacTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, torch::Device device, const SacHyperParams &params);

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

}// namespace arenai::agent

#endif//ARENAI_SAC_FACTORY_H
