//
// Created by claude on 08/11/2026.
//

#ifndef ARENAI_SAC_TRAINER_H
#define ARENAI_SAC_TRAINER_H

#include <memory>
#include <tuple>
#include <vector>

#include "../../networks/actor.h"
#include "../../networks/entropy.h"
#include "../../networks/q_function.h"
#include "../trainer.h"
#include "./sac_replay_buffer.h"

namespace arenai::agent {

    // Soft Actor-Critic over a factorized hybrid policy: tanh-gaussian continuous
    // actions (reparameterized) and independent Bernoulli discrete actions whose
    // expectation is taken exactly over their combinations. One temperature per action
    // (continuous axes first, then discrete), each tuned toward its own entropy target, so
    // that no axis can pay the entropy of another.
    class SacTrainer final : public AbstractTrainer {
    public:
        SacTrainer(
            const std::shared_ptr<Actor> &actor,
            const std::shared_ptr<SacReplayBuffer> &replay_buffer, int nb_sensors,
            int nb_continuous_actions, int nb_discrete_actions, float actor_learning_rate,
            float critic_learning_rate, float alpha_learning_rate, int hidden_size_sensors,
            const std::vector<int> &critic_hidden_sizes,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
            float gamma, float tau, float initial_alpha, float target_sigma,
            float discrete_target_entropy_ratio, int learning_starts, int gradient_steps,
            int batch_size, int gpu_batch_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;
        std::vector<std::shared_ptr<AbstractMetric>> get_csv_only_metrics() override;

        void save(const std::filesystem::path &output_folder) override;

        int count_parameters() override;

    private:
        std::shared_ptr<Actor> actor;
        std::shared_ptr<SacReplayBuffer> replay_buffer;

        std::shared_ptr<TwinQNetwork> critic;
        std::shared_ptr<TwinQNetwork> target_critic;

        // [nb_continuous + nb_discrete] temperatures, continuous axes first
        std::shared_ptr<SacAlpha> alphas;

        std::unique_ptr<torch::optim::Adam> actor_optim;
        std::unique_ptr<torch::optim::Adam> critic_optim;
        std::unique_ptr<torch::optim::Adam> alpha_optim;

        std::shared_ptr<AbstractMetric> critic_loss_metric;
        std::shared_ptr<AbstractMetric> actor_loss_metric;
        // Q of the taken actions, per critic block
        std::shared_ptr<AbstractMetric> q_value_metric;
        std::shared_ptr<AbstractMetric> continuous_entropy_metric;
        std::shared_ptr<AbstractMetric> discrete_entropy_metric;
        // one per temperature, in the alphas' order
        std::vector<std::shared_ptr<AbstractMetric>> alpha_metrics;

        int nb_continuous_actions;
        float gamma;
        float tau;
        // per continuous axis
        float continuous_target_entropy;
        float discrete_target_entropy_ratio;

        int learning_starts;
        int gradient_steps;
        int batch_size;
        int gpu_batch_size;

        void train() const;

        // one optimizer step of each network over a sampled batch, cut into device blocks
        void train_step() const;

        // accumulate a device block's share of the batch gradient
        void backward_critic(const SacBatch &block, float batch_fraction) const;
        void backward_actor_and_alphas(const SacBatch &block, float batch_fraction) const;

        // dual loss of the temperatures over the actor's (detached) entropies
        void backward_alphas(
            const torch::Tensor &continuous_log_prob, const torch::Tensor &discrete_entropies,
            const torch::Tensor &discrete_mask, float batch_fraction) const;

        // (continuous, discrete) detached temperatures
        std::pair<torch::Tensor, torch::Tensor> split_alphas() const;

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_SAC_TRAINER_H
