#ifndef ARENAI_PPG_TRAINER_H
#define ARENAI_PPG_TRAINER_H

#include "../../networks/actor.h"
#include "../../networks/value_function.h"
#include "../../networks_utils/warmup.h"
#include "../entropy_alphas.h"
#include "../return_scaler.h"
#include "../trainer.h"
#include "./ppg_rollout_buffer.h"

namespace arenai::agent {

    struct PpgGaeResult {
        torch::Tensor advantages;
        torch::Tensor returns;
    };

    class PpgTrainer final : public AbstractTrainer {
    public:
        PpgTrainer(
            const std::shared_ptr<DistillationActor> &actor,
            const std::shared_ptr<PpgRolloutBuffer> &rollout_buffer, int nb_continuous_actions,
            int nb_discrete_actions, int nb_sensors, float actor_learning_rate,
            float critic_learning_rate, int hidden_size_sensors,
            const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float gamma, float gae_lambda, float clip_epsilon, float target_kl, float grad_norm_max,
            float initial_alpha, float alpha_learning_rate, float continuous_target_entropy,
            float discrete_target_proba, int actor_epochs, int critic_epochs, int rollout_size,
            int actor_batch_size, int actor_gpu_batch_size, int critic_batch_size,
            int critic_gpu_batch_size, int actor_aux_epochs, int critic_aux_epochs,
            float beta_clone, int n_pi);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;
        std::vector<std::shared_ptr<AbstractMetric>> get_csv_only_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        std::shared_ptr<DistillationActor> actor;
        std::shared_ptr<PpgRolloutBuffer> rollout_buffer;

        // one learned entropy temperature per action
        std::unique_ptr<EntropyAlphas> alphas;

        std::shared_ptr<ValueFunction> critic;

        std::unique_ptr<torch::optim::Adam> actor_optim;
        std::unique_ptr<torch::optim::Adam> critic_optim;

        // rewards scaled by the running std of the discounted return before the GAE
        std::shared_ptr<ReturnScaler> return_scaler;

        std::shared_ptr<AbstractMetric> actor_mean_loss_metric;
        std::shared_ptr<AbstractMetric> critic_mean_loss_metric;

        // share of the return variance the critic explains, per critic minibatch
        std::shared_ptr<AbstractMetric> explained_variance_metric;

        // both regulated by their learned alphas (CSV-only metrics)
        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;

        // recorded on every minibatch
        std::shared_ptr<AbstractMetric> clip_fraction_metric;
        // recorded on every actor optimizer step taken or refused
        std::shared_ptr<AbstractMetric> kl_metric;

        // share of the planned actor steps the KL threshold cut, once per rollout
        std::shared_ptr<AbstractMetric> skip_fraction_metric;

        float gamma;
        float gae_lambda;
        float clip_epsilon;
        float target_kl;

        float grad_norm_max;

        int actor_epochs;
        int critic_epochs;
        int rollout_size;
        int actor_batch_size;
        int actor_gpu_batch_size;
        int critic_batch_size;
        int critic_gpu_batch_size;

        int actor_aux_epochs;
        int critic_aux_epochs;

        float beta_clone;

        // the auxiliary phase runs once every n_pi policy phases
        int n_pi;
        int nb_policy_phases = 0;

        // the rollout flattened to [T * nb_tanks, ...] rows, all on the device
        struct FlatRows {
            torch::Tensor vision;
            torch::Tensor proprioception;
            torch::Tensor continuous_pre_tanh;
            torch::Tensor discrete_actions;
            torch::Tensor fire_masks;
            torch::Tensor old_log_probs;
            torch::Tensor advantages;
            torch::Tensor returns;

            FlatRows select(const torch::Tensor &idx) const;
        };

        // the policy to keep through the auxiliary phase (the clone reference)
        struct CloneTargets {
            torch::Tensor mus;
            torch::Tensor sigmas;
            torch::Tensor discrete_probas;

            CloneTargets select(const torch::Tensor &idx) const;
        };

        void train(bool run_aux) const;

        void run_actor_epochs(const FlatRows &rows, const torch::Tensor &valid_idx) const;
        void run_critic_epochs(
            int nb_epochs, const FlatRows &rows, const torch::Tensor &valid_idx) const;

        // once every n_pi policy phases: value distillation in the actor + extra critic epochs
        void auxiliary_phase(const FlatRows &rows, const torch::Tensor &valid_idx) const;
        CloneTargets compute_clone_targets(const FlatRows &rows) const;
        void run_actor_aux_epochs(
            const FlatRows &rows, const CloneTargets &targets,
            const torch::Tensor &valid_idx) const;

        // accumulates a device block's share of its optimizer batch gradient; returns its per-row KL
        torch::Tensor backward_actor(const FlatRows &block, float batch_fraction) const;

        // value distillation + policy clone loss of the auxiliary phase
        void backward_actor_aux(
            const FlatRows &block, const CloneTargets &targets, float batch_fraction) const;

        void backward_critic(const FlatRows &block, float batch_fraction) const;

        PpgGaeResult compute_gae(const PpgRollout &rollout, torch::Device device) const;

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_PPG_TRAINER_H
