//
// Created by samuel on 22/07/2026.
//

#ifndef ARENAI_PPG_FACTORY_H
#define ARENAI_PPG_FACTORY_H

#include "../torch_factory.h"
#include "./ppg_agent.h"
#include "./ppg_collector.h"
#include "./ppg_hyperparams.h"
#include "./ppg_rollout_buffer.h"
#include "./ppg_trainer.h"

namespace arenai::agent {

    class PpgTorchAgentFactory final : public AbstractTorchAgentFactory {
    public:
        PpgTorchAgentFactory(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, torch::Device device, const PpgHyperParams &params);

        std::shared_ptr<AbstractTorchAgent> get_agent() override;
        std::shared_ptr<AbstractStepCollector> get_collector() override;
        std::shared_ptr<AbstractTrainer> get_trainer() override;

        nlohmann::json get_config() const override;

    private:
        nlohmann::json config;

        // triad built once, sharing actor + rollout_buffer
        std::shared_ptr<DistillationActor> actor;

        std::shared_ptr<PpgRolloutBuffer> rollout_buffer;
        std::shared_ptr<PpgStepCollector> collector;
        std::shared_ptr<TorchPpgAgent> agent;
        std::shared_ptr<PpgTrainer> trainer;
    };

}// namespace arenai::agent

#endif//ARENAI_PPG_FACTORY_H
