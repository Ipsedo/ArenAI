//
// Created by claude on 22/07/2026.
//

#ifndef ARENAI_PPO_HYPERPARAMS_H
#define ARENAI_PPO_HYPERPARAMS_H

#include <tuple>
#include <vector>

#include "../../utils/cli_fields.h"

namespace arenai::agent {

    // Member initializers are the CLI defaults (single source of truth).
    struct PpoHyperParams {
        float actor_learning_rate = 1e-4f;
        float critic_learning_rate = 3e-4f;
        int hidden_size_sensors = 128;
        std::vector<int> actor_hidden_sizes = {1024, 512};
        std::vector<int> critic_hidden_sizes = {1024, 512};
        std::vector<std::tuple<int, int>> vision_channels = {{3, 8},   {8, 16},  {16, 24},
                                                             {24, 32}, {32, 48}, {48, 64}};
        std::vector<int> group_norm_nums = {1, 2, 3, 4, 6, 8};
        float initial_sigma = 0.5f;
        float initial_fire_proba = 0.3f;
        int metric_window_size = 256;
        // 0.99 at 30 Hz -> ~3.3 s bootstrap horizon. 0.997 scaled the returns by 333x the
        // per-step reward for a credit horizon gae_lambda already carries: the critic was
        // left chasing a high-variance target it never caught (train_353: MSE ~0.22, i.e.
        // an RMSE 3x the aim signal reaching the fire decision, so the normalised advantages
        // were mostly critic error)
        float gamma = 0.99f;
        // 0.98: a shell resolving 30-60 steps after the fire still reaches the fire
        // decision at x0.16-0.40 through GAE (gamma * lambda = 0.970), instead of
        // x0.03-0.16 with 0.95. Raise to 0.99 to get the x0.25-0.50 the 0.997 gamma gave
        float gae_lambda = 0.98f;
        float clip_epsilon = 0.2f;
        // early-stop of the epoch loop when approx KL > 1.5 * target_kl; <= 0 disables it.
        // 0.05: the clip is already a trust region, this is the second guard. At 0.02 the
        // threshold (0.03) sat right on the observed KL distribution and became the binding
        // constraint — train_353 skipped 76% of the actor minibatches, and the skip is not a
        // neutral subsample: it drops exactly the updates that would move the policy most
        float target_kl = 0.05f;
        float grad_norm_max = 0.5f;
        float continuous_entropy_coefficient = 2e-3f;
        float discrete_entropy_coefficient = 2e-3f;
        // 2: with 32 tanks x 900 steps / 1024 the rollout is ~29 minibatches, so 4 epochs
        // meant 116 sequential Adam steps against a single old_log_probs snapshot. The KL
        // grows with the steps taken since that snapshot, so the tail epochs were skipped by
        // the actor and applied by the critic alone — 4x the compute for ~1 usable epoch
        int epochs = 2;
        int rollout_size = 30 * 30;
        int minibatch_size = 1024;
    };

    std::vector<CliField<PpoHyperParams>> ppo_cli_fields();

}// namespace arenai::agent

#endif//ARENAI_PPO_HYPERPARAMS_H
