//
// Created by samuel on 21/01/2026.
//

#ifndef ARENAI_TRAIN_HOST_SAC_H
#define ARENAI_TRAIN_HOST_SAC_H

#include "../../include/arenai_train/agent.h"
#include "../networks/actor.h"
#include "../networks/entropy.h"
#include "../networks/q_function.h"

namespace arenai::train {

    class SacAgent : public AbstractAgent {
    public:
        SacAgent(
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            int nb_discrete_actions, float actor_learning_rate, float critic_learning_rate,
            float alpha_learning_rate, int hidden_size_sensors, int hidden_size_actions,
            const std::vector<int> &actor_hidden_sizes, const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float tau, float gamma);

        void train(const std::unique_ptr<ReplayBuffer> &replay_buffer, int epochs, int batch_size)
            override;
        agent_response act(const torch::Tensor &vision, const torch::Tensor &sensors) override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;

        void save(const std::filesystem::path &output_folder) override;
        void load(const std::filesystem::path &input_folder) override;

        void set_train(bool train) override;
        void to(torch::Device device) override;

        int count_parameters() override;

    private:
        static constexpr int WARMUP_STEP = 10000;

        std::shared_ptr<Actor> actor;

        std::shared_ptr<QFunction> critic_1;
        std::shared_ptr<QFunction> critic_2;

        std::shared_ptr<QFunction> target_critic_1;
        std::shared_ptr<QFunction> target_critic_2;

        std::shared_ptr<AlphaParameter> alpha_continuous;
        std::shared_ptr<AlphaParameter> alpha_discrete;

        std::shared_ptr<TargetEntropyWarmup> continuous_target_entropy;
        float _discrete_maximal_entropy;
        std::shared_ptr<TargetEntropyWarmup> discrete_target_entropy;

        std::shared_ptr<torch::optim::Adam> actor_optim;
        std::shared_ptr<torch::optim::Adam> critic_1_optim;
        std::shared_ptr<torch::optim::Adam> critic_2_optim;

        std::shared_ptr<torch::optim::Adam> alpha_continuous_optim;
        std::shared_ptr<torch::optim::Adam> alpha_discrete_optim;

        std::shared_ptr<AbstractMetric> actor_loss_metric;

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
    };

}// namespace arenai::train

#endif//ARENAI_TRAIN_HOST_SAC_H
