//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_LIQUID_PPO_TRAINER_H
#define ARENAI_LIQUID_PPO_TRAINER_H

#include "../../networks/entropy.h"
#include "../../networks/recurrent/liquid_actor.h"
#include "../../networks/recurrent/liquid_critic.h"
#include "../trainer.h"
#include "./liquid_ppo_rollout_buffer.h"

namespace arenai::agent {

    struct LiquidGaeResult {
        torch::Tensor advantages;
        torch::Tensor returns;
        // [T, nb_tanks, neuron_number] critic liquid state opening each step
        torch::Tensor critic_hiddens;
    };

    // Recurrent PPO: the updates run on contiguous [chunk_size]-long sequences
    // (truncated BPTT), each initialized with the liquid state recorded when the
    // chunk's first step was collected.
    class LiquidPpoTrainer final : public AbstractTrainer {
    public:
        LiquidPpoTrainer(
            const std::shared_ptr<LiquidActor> &actor,
            const std::shared_ptr<LiquidPpoRolloutBuffer> &rollout_buffer, int vision_height,
            int vision_width, int nb_sensors, int nb_continuous_actions, int nb_discrete_action,
            float actor_learning_rate, float critic_learning_rate, int hidden_size_sensors,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, int neuron_number, int unfolding_steps,
            float delta_t, torch::Device device, int metric_window_size, float gamma,
            float gae_lambda, float clip_epsilon, float target_kl, float grad_norm_max,
            float continuous_target_entropy, float discrete_target_entropy_factor, int epochs,
            int rollout_size, int minibatch_size, int chunk_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        std::shared_ptr<LiquidActor> actor;
        std::shared_ptr<LiquidPpoRolloutBuffer> rollout_buffer;

        std::unique_ptr<PidLagrangianAlphaParameters> continuous_alpha;
        std::unique_ptr<PidLagrangianAlphaParameters> discrete_alpha;

        float continuous_target_entropy;
        float discrete_target_entropy;

        std::shared_ptr<LiquidCritic> critic;

        std::unique_ptr<torch::optim::Adam> actor_optim;
        std::unique_ptr<torch::optim::Adam> critic_optim;

        std::shared_ptr<AbstractMetric> actor_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_mean_loss_metric;

        // share of the return variance the critic explains, per critic minibatch
        std::shared_ptr<AbstractMetric> explained_variance_metric;

        // both regulated by their constant entropy bonus
        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;

        std::shared_ptr<AbstractMetric> continuous_alpha_metric;
        std::shared_ptr<AbstractMetric> discrete_alpha_metric;

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
        int chunk_size;

        // critic liquid state opening the next rollout's first step, carried
        // across train() calls; re-drawn at every episode start
        torch::Tensor critic_carry;

        void train();

        bool train_actor(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &hidden, const torch::Tensor &continuous_actions,
            const torch::Tensor &discrete_actions, const torch::Tensor &old_log_probs,
            const torch::Tensor &advantages, const torch::Tensor &valids) const;

        void train_critic(
            const torch::Tensor &vision, const torch::Tensor &proprioception,
            const torch::Tensor &hidden, const torch::Tensor &returns,
            const torch::Tensor &valids) const;

        // sequential critic pass over the rollout, advancing critic_carry
        LiquidGaeResult compute_gae(const LiquidPpoRollout &rollout, torch::Device device);

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_TRAINER_H
