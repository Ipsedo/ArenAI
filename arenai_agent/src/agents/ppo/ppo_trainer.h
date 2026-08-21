//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_TRAINER_H
#define ARENAI_PPO_TRAINER_H

#include "../../networks/actor.h"
#include "../../networks/entropy.h"
#include "../../networks/value_function.h"
#include "../trainer.h"
#include "./ppo_rollout_buffer.h"

namespace arenai::agent {

    struct GaeResult {
        torch::Tensor advantages;
        torch::Tensor returns;
    };

    class PpoTrainer final : public AbstractTrainer {
    public:
        PpoTrainer(
            const std::shared_ptr<Actor> &actor,
            const std::shared_ptr<PpoRolloutBuffer> &rollout_buffer, int vision_height,
            int vision_width, int nb_sensors, int nb_continuous_actions, float actor_learning_rate,
            float critic_learning_rate, float alpha_learning_rate, int hidden_size_sensors,
            const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float gamma, float gae_lambda, float clip_epsilon, float target_kl, float grad_norm_max,
            float target_sigma, float target_fire_proba, int epochs, int rollout_size,
            int minibatch_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        std::shared_ptr<Actor> actor;
        std::shared_ptr<PpoRolloutBuffer> rollout_buffer;

        std::shared_ptr<PidLagrangianAlphaParameters> continuous_alpha;
        std::shared_ptr<PidLagrangianAlphaParameters> discrete_alpha;

        std::shared_ptr<ValueFunction> critic;

        std::unique_ptr<torch::optim::Adam> actor_optim;
        std::unique_ptr<torch::optim::Adam> critic_optim;

        std::shared_ptr<AbstractMetric> actor_mean_loss_metric;
        std::shared_ptr<AbstractMetric> actor_std_loss_metric;

        std::shared_ptr<AbstractMetric> critic_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_std_loss_metric;

        // share of the return variance the critic explains, per critic minibatch
        std::shared_ptr<AbstractMetric> explained_variance_metric;

        // both regulated by their constant entropy bonus
        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;

        std::shared_ptr<AbstractMetric> continuous_alpha_metric;
        std::shared_ptr<AbstractMetric> discrete_alpha_metric;

        // mean sigma of the truncated normal: direct view of the aim spread, Hc only bounds it
        std::shared_ptr<AbstractMetric> sigma_metric;

        // both recorded on every attempted minibatch, skipped ones included
        std::shared_ptr<AbstractMetric> clip_fraction_metric;
        std::shared_ptr<AbstractMetric> kl_metric;

        // fraction of minibatches the KL threshold skipped
        std::shared_ptr<AbstractMetric> skip_fraction_metric;

        float gamma;
        float gae_lambda;
        float clip_epsilon;
        // approx-KL threshold ending the epoch loop early (<= 0: disabled)
        float target_kl;

        float grad_norm_max;

        float target_sigma;
        float target_fire_proba;

        int epochs;
        // rollout horizon: one train() consumes rollout_size complete steps in a single update
        int rollout_size;
        // rows per gradient step; also the chunk size for the no-grad value evaluation
        int minibatch_size;

        void train() const;

        // one backward pass on the actor for a single minibatch; returns true when the
        // minibatch drifted past the KL threshold, in which case no update was applied and
        // only the kl metric was recorded — the next minibatch is tried either way
        bool train_actor(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions,
            const torch::Tensor &old_log_probs, const torch::Tensor &advantages) const;

        // one backward pass on the critic for a single minibatch
        void train_critic(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &returns) const;

        // GAE advantages (normalized over the valid pairs) and value targets,
        // computed with the pre-update critic
        GaeResult compute_gae(const PpoRollout &rollout, torch::Device device) const;

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_TRAINER_H
