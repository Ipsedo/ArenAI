//
// Created by claude on 26/10/2026.
//

#ifndef ARENAI_IMPALA_PPO_TRAINER_H
#define ARENAI_IMPALA_PPO_TRAINER_H

#include "../../networks/impala_actor.h"
#include "../../networks/impala_value_function.h"
#include "../../networks_utils/warmup.h"
#include "../ppo/ppo_rollout_buffer.h"
#include "../return_scaler.h"
#include "../trainer.h"

namespace arenai::agent {

    struct ImpalaGaeResult {
        torch::Tensor advantages;
        torch::Tensor returns;
    };

    // PpoTrainer with the IMPALA actor / critic pair; same update, same metrics
    class ImpalaPpoTrainer final : public AbstractTrainer {
    public:
        ImpalaPpoTrainer(
            const std::shared_ptr<ImpalaActor> &actor,
            const std::shared_ptr<PpoRolloutBuffer> &rollout_buffer, int nb_sensors,
            float actor_learning_rate, float critic_learning_rate, int hidden_size_sensors,
            const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels, torch::Device device,
            int metric_window_size, float gamma, float gae_lambda, float clip_epsilon,
            float target_kl, float grad_norm_max, float continuous_entropy_factor,
            float discrete_entropy_factor, int epochs, int rollout_size, int minibatch_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        std::shared_ptr<ImpalaActor> actor;
        std::shared_ptr<PpoRolloutBuffer> rollout_buffer;

        float continuous_entropy_factor;
        float discrete_entropy_factor;

        std::shared_ptr<ImpalaValueFunction> critic;

        std::unique_ptr<torch::optim::Adam> actor_optim;
        std::unique_ptr<torch::optim::Adam> critic_optim;

        // rewards scaled by the running std of the discounted return before the GAE
        std::shared_ptr<ReturnScaler> return_scaler;

        std::shared_ptr<AbstractMetric> actor_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_mean_loss_metric;

        // share of the return variance the critic explains, per critic minibatch
        std::shared_ptr<AbstractMetric> explained_variance_metric;

        // both regulated by their constant entropy bonus
        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;

        // both recorded on every attempted minibatch, skipped ones included
        std::shared_ptr<AbstractMetric> clip_fraction_metric;
        std::shared_ptr<AbstractMetric> kl_metric;

        // fraction of minibatches the KL threshold skipped
        std::shared_ptr<AbstractMetric> skip_fraction_metric;

        float gamma;
        float gae_lambda;
        float clip_epsilon;
        float target_kl;

        float grad_norm_max;

        int epochs;
        int rollout_size;
        int minibatch_size;

        void train();

        bool train_actor(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions,
            const torch::Tensor &fire_masks, const torch::Tensor &old_log_probs,
            const torch::Tensor &advantages) const;

        void train_critic(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &returns) const;

        ImpalaGaeResult compute_gae(const PpoRollout &rollout, torch::Device device) const;

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_IMPALA_PPO_TRAINER_H
