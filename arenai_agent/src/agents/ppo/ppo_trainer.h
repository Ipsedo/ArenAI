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

    struct ActorUpdateResult {
        // the minibatch drifted past the KL threshold: no update was applied
        bool kl_exceeded;
        // one entropy per continuous action, detached: each dimension feeds its own alpha
        torch::Tensor continuous_entropy;
        float discrete_entropy;
    };

    class PpoTrainer final : public AbstractTrainer {
    public:
        PpoTrainer(
            std::shared_ptr<Actor> actor, std::shared_ptr<PpoRolloutBuffer> rollout_buffer,
            int vision_height, int vision_width, int nb_sensors, int nb_continuous_actions,
            float actor_learning_rate, float critic_learning_rate,
            float continuous_alpha_learning_rate, float discrete_alpha_learning_rate,
            int hidden_size_sensors, const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float gamma, float gae_lambda, float clip_epsilon, float target_kl, float grad_norm_max,
            int epochs, int rollout_size, int minibatch_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        std::shared_ptr<Actor> actor;
        std::shared_ptr<PpoRolloutBuffer> rollout_buffer;

        std::shared_ptr<ValueFunction> critic;

        // adaptive entropy coefficients (dual ascent toward fixed entropy targets)
        std::shared_ptr<RangeAlphaParameters> alpha_continuous;
        std::shared_ptr<RangeAlphaParameters> alpha_discrete;
        std::shared_ptr<AbstractTargetEntropy> continuous_target_entropy;
        std::shared_ptr<AbstractTargetEntropy> discrete_target_entropy;

        std::shared_ptr<torch::optim::Adam> actor_optim;
        std::shared_ptr<torch::optim::Adam> critic_optim;
        std::shared_ptr<torch::optim::SGD> alpha_continuous_optim;
        std::shared_ptr<torch::optim::SGD> alpha_discrete_optim;

        std::shared_ptr<AbstractMetric> actor_mean_loss_metric;
        std::shared_ptr<AbstractMetric> actor_std_loss_metric;

        std::shared_ptr<AbstractMetric> critic_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_std_loss_metric;

        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;

        std::shared_ptr<AbstractMetric> alpha_continuous_metric;
        // alpha of the most constrained dimension: it is the one that rises when a single
        // action collapses, which the mean hides
        std::shared_ptr<AbstractMetric> alpha_continuous_max_metric;
        std::shared_ptr<AbstractMetric> alpha_discrete_metric;

        std::shared_ptr<AbstractMetric> continuous_target_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_target_entropy_metric;

        // mean sigma of the truncated normal: direct view of the aim spread, Hc only bounds it
        std::shared_ptr<AbstractMetric> sigma_metric;

        // sigma of the tightest action dimension: the entropy target constrains the sum over
        // the dimensions, so a single wide one can hide several collapsed ones behind a healthy
        // Hc — and a collapsed dimension is what blows the log ratio up
        std::shared_ptr<AbstractMetric> sigma_min_metric;

        std::shared_ptr<AbstractMetric> clip_fraction_metric;
        std::shared_ptr<AbstractMetric> kl_metric;

        float gamma;
        float gae_lambda;
        float clip_epsilon;
        // approx-KL threshold ending the epoch loop early (<= 0: disabled)
        float target_kl;

        float grad_norm_max;

        int epochs;
        // rollout horizon: one train() consumes rollout_size complete steps in a single update
        int rollout_size;
        // rows per gradient step; also the chunk size for the no-grad value evaluation
        int minibatch_size;

        void train() const;

        // one backward pass on the actor for a single minibatch; the update is skipped
        // when the minibatch already drifted past the KL threshold, its entropies are
        // reported either way so the caller can feed the dual ascent
        ActorUpdateResult train_actor(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions,
            const torch::Tensor &old_log_probs, const torch::Tensor &advantages) const;

        // one backward pass on the critic for a single minibatch
        void train_critic(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &returns) const;

        // one backward pass of the dual ascent, fed with the mean entropy of the whole update:
        // a scalar for the discrete coefficient, one entropy per action for the continuous ones
        static void train_alpha(
            const std::shared_ptr<RangeAlphaParameters> &alpha,
            const std::shared_ptr<torch::optim::SGD> &optim,
            const std::shared_ptr<AbstractTargetEntropy> &target_entropy,
            const torch::Tensor &mean_entropy);

        // GAE advantages (normalized over the valid pairs) and value targets,
        // computed with the pre-update critic
        GaeResult compute_gae(const PpoRollout &rollout, torch::Device device) const;

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_PPO_TRAINER_H
