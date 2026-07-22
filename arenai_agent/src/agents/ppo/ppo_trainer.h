//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_TRAINER_H
#define ARENAI_PPO_TRAINER_H

#include "../../networks/actor.h"
#include "../../networks/value_function.h"
#include "../trainer.h"
#include "./ppo_rollout_buffer.h"

namespace arenai::agent {

    class PpoTrainer final : public AbstractTrainer {
    public:
        PpoTrainer(
            std::shared_ptr<Actor> actor, std::shared_ptr<PpoRolloutBuffer> rollout_buffer,
            int vision_height, int vision_width, int nb_sensors, float actor_learning_rate,
            float critic_learning_rate, int hidden_size_sensors,
            const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float gamma, float gae_lambda, float clip_epsilon, float continuous_entropy_coef,
            float discrete_entropy_coef, int epochs, int rollout_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        static constexpr double GRAD_NORM_MAX = 1.0;

        std::shared_ptr<Actor> actor;
        std::shared_ptr<PpoRolloutBuffer> rollout_buffer;

        std::shared_ptr<ValueFunction> critic;

        std::shared_ptr<torch::optim::Adam> actor_optim;
        std::shared_ptr<torch::optim::Adam> critic_optim;

        std::shared_ptr<AbstractMetric> actor_mean_loss_metric;
        std::shared_ptr<AbstractMetric> actor_std_loss_metric;

        std::shared_ptr<AbstractMetric> critic_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_std_loss_metric;

        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;

        std::shared_ptr<AbstractMetric> clip_fraction_metric;

        float gamma;
        float gae_lambda;
        float clip_epsilon;

        float continuous_entropy_coef;
        float discrete_entropy_coef;

        int epochs;
        // rollout horizon: one train() consumes rollout_size complete steps in a single update
        int rollout_size;

        void train() const;

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_TRAINER_H
