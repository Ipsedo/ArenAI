//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_SAC_TRAINER_H
#define ARENAI_SAC_TRAINER_H

#include "../../networks/actor.h"
#include "../../networks/entropy.h"
#include "../../networks/q_function.h"
#include "../trainer.h"
#include "./replay_buffer.h"

namespace arenai::train {

    class SacTrainer final : public AbstractTrainer {
    public:
        SacTrainer(
            std::shared_ptr<Actor> actor, std::shared_ptr<SacReplayBuffer> replay_buffer,
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, float actor_learning_rate, float critic_learning_rate,
            float alpha_learning_rate, int hidden_size_sensors, int hidden_size_actions,
            const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float tau, float gamma, int train_every, int epochs, int batch_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        // FPS * sec * warmup_episodes / train_every
        static constexpr int WARMUP_STEP = 30 * 30 * 500 / 64;

        static constexpr double GRAD_NORM_MAX = 1.0;

        std::shared_ptr<Actor> actor;
        std::shared_ptr<SacReplayBuffer> replay_buffer;

        std::shared_ptr<QFunction> critic_1;
        std::shared_ptr<QFunction> critic_2;

        std::shared_ptr<QFunction> target_critic_1;
        std::shared_ptr<QFunction> target_critic_2;

        std::shared_ptr<AlphaParameter> alpha_continuous;
        std::shared_ptr<AlphaParameter> alpha_discrete;

        std::shared_ptr<AbstractTargetEntropy> continuous_target_entropy;
        std::shared_ptr<AbstractTargetEntropy> discrete_target_entropy;

        std::shared_ptr<torch::optim::Adam> actor_optim;
        std::shared_ptr<torch::optim::Adam> critic_1_optim;
        std::shared_ptr<torch::optim::Adam> critic_2_optim;

        std::shared_ptr<torch::optim::Adam> alpha_continuous_optim;
        std::shared_ptr<torch::optim::Adam> alpha_discrete_optim;

        std::shared_ptr<AbstractMetric> actor_mean_loss_metric;
        std::shared_ptr<AbstractMetric> actor_std_loss_metric;

        std::shared_ptr<AbstractMetric> critic_1_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_1_std_loss_metric;

        std::shared_ptr<AbstractMetric> critic_2_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_2_std_loss_metric;

        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;

        std::shared_ptr<AbstractMetric> alpha_continuous_metric;
        std::shared_ptr<AbstractMetric> alpha_discrete_metric;

        std::shared_ptr<AbstractMetric> continuous_target_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_target_entropy_metric;

        float tau;
        float gamma;

        int train_every;
        int train_counter;

        int epochs;
        int batch_size;

        void train() const;

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::train

#endif//ARENAI_SAC_TRAINER_H
