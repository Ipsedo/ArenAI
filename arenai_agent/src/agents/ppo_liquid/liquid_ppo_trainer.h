#ifndef ARENAI_LIQUID_PPO_TRAINER_H
#define ARENAI_LIQUID_PPO_TRAINER_H

#include "../../networks/entropy.h"
#include "../../networks/recurrent/liquid_actor.h"
#include "../../networks/recurrent/liquid_critic.h"
#include "../entropy_alphas.h"
#include "../return_scaler.h"
#include "../trainer.h"
#include "./liquid_ppo_rollout_buffer.h"

namespace arenai::agent {

    struct LiquidGaeResult {
        torch::Tensor advantages;
        torch::Tensor returns;
        // [T, nb_tanks, neuron_number] critic liquid state opening each step
        torch::Tensor critic_hiddens;
    };

    // per live transition of a rollout, the two first-epoch pushes on log sigma
    struct LiquidSigmaGradients {
        // [N] indices in the rollout
        torch::Tensor steps;
        torch::Tensor tanks;
        // [N, nb_continuous_actions] d(surrogate)/d(log sigma), d(alpha * H)/d(log sigma)
        torch::Tensor surrogate;
        torch::Tensor entropy;
        torch::Tensor mu;
        torch::Tensor sigma;
        // [N]
        torch::Tensor advantages;
        torch::Tensor fire_masks;
        torch::Tensor fires;
    };

    class LiquidPpoTrainer final : public AbstractTrainer {
    public:
        LiquidPpoTrainer(
            const std::shared_ptr<LiquidActor> &actor,
            const std::shared_ptr<LiquidPpoRolloutBuffer> &rollout_buffer,
            int nb_continuous_actions, int nb_discrete_actions, int nb_sensors,
            float actor_learning_rate, float critic_learning_rate, int hidden_size_sensors,
            const std::vector<std::tuple<int, int>> &vision_channels,
            const std::vector<int> &group_norm_nums, int neuron_number, int unfolding_steps,
            float delta_t, torch::Device device, int metric_window_size, float gamma,
            float gae_lambda, float clip_epsilon, float target_kl, float grad_norm_max,
            float initial_alpha, float alpha_learning_rate,
            const std::vector<float> &continuous_target_entropy, float discrete_target_proba,
            int actor_epochs, int critic_epochs, int rollout_size, int actor_batch_size,
            int actor_gpu_batch_size, int critic_batch_size, int critic_gpu_batch_size,
            int chunk_size);

        void step() override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() override;
        std::vector<std::shared_ptr<AbstractMetric>> get_csv_only_metrics() override;

        void save(const std::filesystem::path &output_folder) override;
        void load(const std::filesystem::path &input_folder) const;

        // consumes the complete steps of the buffer without any optimizer step
        LiquidSigmaGradients sigma_gradients();

        int count_parameters() override;

    private:
        std::shared_ptr<LiquidActor> actor;
        std::shared_ptr<LiquidPpoRolloutBuffer> rollout_buffer;

        // one learned entropy temperature per action
        std::unique_ptr<EntropyAlphas> alphas;

        std::shared_ptr<LiquidCritic> critic;

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
        int chunk_size;

        torch::Tensor critic_carry;

        // the kept steps flattened to [T_kept * nb_tanks, ...] rows, all on the device
        struct FlatRows {
            torch::Tensor vision;
            torch::Tensor proprioception;
            torch::Tensor continuous_pre_tanh;
            torch::Tensor discrete_actions;
            torch::Tensor fire_masks;
            torch::Tensor old_log_probs;
            torch::Tensor advantages;
            torch::Tensor returns;
            torch::Tensor valids;
            torch::Tensor actor_hiddens;
            torch::Tensor critic_hiddens;

            FlatRows select_chunks(
                const torch::Tensor &rows, const torch::Tensor &start_rows,
                int64_t chunk_size) const;
        };

        struct ChunkIndex {
            int64_t nb_tanks;
            int64_t chunk_size;
            torch::Device device;
            torch::Tensor step_offsets;
            // live transitions per chunk unit
            torch::Tensor valid_counts;
            // units with at least one live transition: the only ones worth training on
            torch::Tensor valid_units;

            ChunkIndex(
                const torch::Tensor &valids, int64_t nb_tanks, int64_t nb_chunks_per_tank,
                int64_t chunk_size, torch::Device device);

            int64_t units_per_batch(int64_t batch_size) const;

            std::tuple<torch::Tensor, torch::Tensor> rows(const torch::Tensor &units) const;

            // live rows of the units, the weight of a block in its optimizer batch
            float live_rows(const torch::Tensor &units) const;
            float batch_live_rows(const std::vector<torch::Tensor> &blocks) const;
        };

        struct ActorTerms {
            // live rows of the block, in its [units, chunk_size] flattening
            torch::Tensor valid_idx;
            torch::Tensor mu;
            torch::Tensor sigma;
            torch::Tensor discrete_mask;
            torch::Tensor ratio;
            torch::Tensor log_ratio;
            torch::Tensor continuous_entropy;
            torch::Tensor discrete_entropy;
            // [N, 1]
            torch::Tensor surrogate;
            // [N, nb_continuous_actions]
            torch::Tensor continuous_bonus;
            torch::Tensor discrete_bonus;
        };

        ActorTerms actor_terms(const FlatRows &block) const;

        static FlatRows flat_rows(
            const LiquidPpoRollout &rollout, const LiquidGaeResult &gae, int64_t nb_kept_steps,
            torch::Device device);

        void train();

        void run_actor_epochs(const FlatRows &rows, const ChunkIndex &chunks) const;
        void run_critic_epochs(const FlatRows &rows, const ChunkIndex &chunks) const;

        // accumulates a device block's share of its optimizer batch gradient; returns its per-row KL
        torch::Tensor backward_actor(const FlatRows &block, float batch_fraction) const;

        void backward_critic(const FlatRows &block, float batch_fraction) const;

        // sequential critic pass over the rollout, advancing critic_carry
        LiquidGaeResult compute_gae(const LiquidPpoRollout &rollout, torch::Device device);

        void set_train(bool train) const;
        void to(torch::Device device) const;
    };

}// namespace arenai::agent

#endif//ARENAI_LIQUID_PPO_TRAINER_H
