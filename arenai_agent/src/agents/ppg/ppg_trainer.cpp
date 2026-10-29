//
// Created by samuel on 22/07/2026.
//

#include "./ppg_trainer.h"

#include <algorithm>
#include <fstream>
#include <numeric>

#include "../../distributions/bernoulli.h"
#include "../../distributions/truncated_normal.h"
#include "../../metrics/mean_metric.h"
#include "../../networks/constants.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/torch_converter.h"
#include "../../networks_utils/torch_saver.h"
#include "../update_utils.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {
    namespace {
        // merges the [T, nb_tanks] leading dimensions into a single row dimension
        torch::Tensor flatten_steps(const torch::Tensor &tensor) {
            auto sizes = tensor.sizes().vec();
            sizes.erase(sizes.begin());
            sizes[0] = tensor.size(0) * tensor.size(1);
            return tensor.reshape(sizes);
        }

        constexpr float LOG_RATIO_MAX_ABS = 3.f;

    }// namespace

    PpgTrainer::PpgTrainer(
        const std::shared_ptr<DistillationActor> &actor,
        const std::shared_ptr<PpgRolloutBuffer> &rollout_buffer, const int nb_sensors,
        const float actor_learning_rate, const float critic_learning_rate,
        const int hidden_size_sensors, const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float gamma, const float gae_lambda,
        const float clip_epsilon, const float target_kl, const float grad_norm_max,
        const float continuous_entropy_factor, const float discrete_entropy_factor,
        const int actor_epochs, const int critic_epochs, const int rollout_size,
        const int actor_batch_size, const int actor_gpu_batch_size, const int critic_batch_size,
        const int critic_gpu_batch_size, const int actor_aux_epochs, const int critic_aux_epochs,
        float beta_clone)
        : actor(actor), rollout_buffer(rollout_buffer),
          continuous_entropy_factor(continuous_entropy_factor),
          discrete_entropy_factor(discrete_entropy_factor),
          critic(std::make_shared<ValueFunction>(
              nb_sensors, hidden_size_sensors, critic_hidden_sizes, vision_channels,
              group_norm_nums)),
          actor_optim(
              std::make_unique<torch::optim::Adam>(this->actor->parameters(), actor_learning_rate)),
          critic_optim(
              std::make_unique<torch::optim::Adam>(critic->parameters(), critic_learning_rate)),
          return_scaler(std::make_shared<ReturnScaler>(gamma)),
          actor_mean_loss_metric(std::make_shared<MeanMetric>("π", metric_window_size)),
          critic_mean_loss_metric(std::make_shared<MeanMetric>("v", metric_window_size)),
          explained_variance_metric(std::make_shared<MeanMetric>("ev", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          clip_fraction_metric(std::make_shared<MeanMetric>("clip", metric_window_size)),
          kl_metric(std::make_shared<MeanMetric>("kl", metric_window_size, 2, true)),
          skip_fraction_metric(std::make_shared<MeanMetric>("skip", metric_window_size)),
          gamma(gamma), gae_lambda(gae_lambda), clip_epsilon(clip_epsilon), target_kl(target_kl),
          grad_norm_max(grad_norm_max), actor_epochs(actor_epochs), critic_epochs(critic_epochs),
          rollout_size(rollout_size), actor_batch_size(actor_batch_size),
          actor_gpu_batch_size(actor_gpu_batch_size), critic_batch_size(critic_batch_size),
          critic_gpu_batch_size(critic_gpu_batch_size), actor_aux_epochs(actor_aux_epochs),
          critic_aux_epochs(critic_aux_epochs), beta_clone(beta_clone) {
        to(device);

        set_train(false);
    }

    void PpgTrainer::step() {
        if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(rollout_size)) train();
    }

    void PpgTrainer::train() const {
        const auto device = actor->parameters().back().device();

        auto rollout = rollout_buffer->get_rollout();

        // the observations go to the device once, for the GAE and every epoch: a
        // per-block transfer would resend the whole rollout's visions on each epoch
        rollout.states.vision = rollout.states.vision.to(device);
        rollout.states.proprioception = rollout.states.proprioception.to(device);

        set_train(false);
        const auto [advantages, returns] = compute_gae(rollout, device);

        set_train(true);

        // flat [T * nb_tanks, ...] rows, all on the device
        const auto flat = [&](const torch::Tensor &tensor) {
            return flatten_steps(tensor).to(device);
        };

        const auto flat_vision = flat(rollout.states.vision);
        const auto flat_proprioception = flat(rollout.states.proprioception);
        const auto flat_continuous_actions = flat(rollout.actions.continuous_action);
        const auto flat_discrete_actions = flat(rollout.actions.discrete_action);
        const auto flat_fire_masks = flat(rollout.states.fire_mask);
        const auto flat_old_log_probs =
            flat(rollout.continuous_log_probs) + flat(rollout.discrete_log_probs);
        const auto flat_advantages = flat(advantages);
        const auto flat_returns = flat(returns);

        // live (step, tank) pairs; the optimizer batches split these rows only
        const auto valid_idx =
            torch::nonzero(flatten_steps(rollout.valids).squeeze(-1)).squeeze(-1).to(device);
        if (const auto nb_valid_rows = valid_idx.size(0); nb_valid_rows == 0) return;

        const auto select = [](const torch::Tensor &tensor, const torch::Tensor &idx) {
            return tensor.index_select(0, idx);
        };

        // Phase #1

        // the actor steps until its KL leaves the trust region: past the threshold it stays
        // for the rest of the rollout
        int nb_actor_steps = 0;
        int nb_planned_actor_steps = 0;
        bool kl_exceeded = false;

        for (int e = 0; e < actor_epochs; e++) {
            const auto batches =
                shuffled_batches(valid_idx, actor_batch_size, actor_gpu_batch_size);
            nb_planned_actor_steps += static_cast<int>(batches.size());

            for (const auto &blocks: batches) {
                if (kl_exceeded) break;

                actor_optim->zero_grad();

                const auto nb_batch_rows = std::accumulate(
                    blocks.begin(), blocks.end(), int64_t{0},
                    [](const int64_t sum, const torch::Tensor &idx) { return sum + idx.size(0); });

                std::vector<torch::Tensor> kl_per_rows;
                for (const auto &idx: blocks)
                    kl_per_rows.push_back(backward_actor(
                        select(flat_vision, idx), select(flat_proprioception, idx),
                        select(flat_continuous_actions, idx), select(flat_discrete_actions, idx),
                        select(flat_fire_masks, idx), select(flat_old_log_probs, idx),
                        select(flat_advantages, idx),
                        static_cast<float>(idx.size(0)) / static_cast<float>(nb_batch_rows)));

                // KL of the policy this step starts from
                const auto approx_kl = trimmed_kl(kl_per_rows);
                kl_metric->add(approx_kl);

                kl_exceeded = target_kl > 0.f && approx_kl > 1.5f * target_kl;
                if (kl_exceeded) break;

                torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
                actor_optim->step();
                nb_actor_steps++;
            }
        }

        skip_fraction_metric->add(
            1.f - static_cast<float>(nb_actor_steps) / static_cast<float>(nb_planned_actor_steps));

        for (int e = 0; e < critic_epochs; e++)
            for (const auto &blocks:
                 shuffled_batches(valid_idx, critic_batch_size, critic_gpu_batch_size)) {
                critic_optim->zero_grad();

                const auto nb_batch_rows = std::accumulate(
                    blocks.begin(), blocks.end(), int64_t{0},
                    [](const int64_t sum, const torch::Tensor &idx) { return sum + idx.size(0); });

                for (const auto &idx: blocks)
                    backward_critic(
                        select(flat_vision, idx), select(flat_proprioception, idx),
                        select(flat_returns, idx),
                        static_cast<float>(idx.size(0)) / static_cast<float>(nb_batch_rows));

                torch::nn::utils::clip_grad_norm_(critic->parameters(), grad_norm_max);
                critic_optim->step();
            }

        /////////
        // PPG //
        /////////

        // Phase #2

        set_train(false);

        // targets of the auxiliary phase, computed once over every flat row before any update:
        // the critic's values to distill and the policy to keep (the clone reference)
        std::vector<torch::Tensor> old_mu_chunks;
        std::vector<torch::Tensor> old_sigma_chunks;
        std::vector<torch::Tensor> old_discrete_chunks;

        {
            torch::NoGradGuard no_grad;

            for (int64_t i = 0; i < flat_vision.size(0); i += critic_gpu_batch_size) {
                const auto end = std::min<int64_t>(i + critic_gpu_batch_size, flat_vision.size(0));

                const auto chunk_vision = flat_vision.slice(0, i, end);
                const auto chunk_proprioception = flat_proprioception.slice(0, i, end);

                const auto [mu, sigma, discrete_proba] =
                    actor->act(chunk_vision, chunk_proprioception);
                old_mu_chunks.push_back(mu);
                old_sigma_chunks.push_back(sigma);
                old_discrete_chunks.push_back(discrete_proba);
            }
        }

        const auto old_mus = torch::cat(old_mu_chunks, 0);
        const auto old_sigmas = torch::cat(old_sigma_chunks, 0);
        const auto old_discretes = torch::cat(old_discrete_chunks, 0);

        // the truncated normal KL has no closed form: Monte-Carlo estimate on actions drawn
        // from the clone reference (the rollout actions come from the pre-phase-1 policy)
        const auto old_continuous_actions = truncated_normal_sample(old_mus, old_sigmas);
        const auto old_continuous_log_probs =
            truncated_normal_log_pdf(old_continuous_actions, old_mus, old_sigmas).sum(-1, true);

        set_train(true);

        // value distillation in the actor, its policy held by the clone term
        for (int e = 0; e < actor_aux_epochs; e++)
            for (const auto &blocks:
                 shuffled_batches(valid_idx, actor_batch_size, actor_gpu_batch_size)) {
                actor_optim->zero_grad();

                const auto nb_batch_rows = std::accumulate(
                    blocks.begin(), blocks.end(), int64_t{0},
                    [](const int64_t sum, const torch::Tensor &idx) { return sum + idx.size(0); });

                for (const auto &idx: blocks)
                    backward_actor_aux(
                        select(flat_vision, idx), select(flat_proprioception, idx),
                        select(flat_fire_masks, idx), select(flat_returns, idx),
                        select(old_continuous_actions, idx), select(old_continuous_log_probs, idx),
                        select(old_discretes, idx),
                        static_cast<float>(idx.size(0)) / static_cast<float>(nb_batch_rows));

                torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
                actor_optim->step();
            }

        // train critic

        for (int e = 0; e < critic_aux_epochs; e++)
            for (const auto &blocks:
                 shuffled_batches(valid_idx, critic_batch_size, critic_gpu_batch_size)) {
                critic_optim->zero_grad();

                const auto nb_batch_rows = std::accumulate(
                    blocks.begin(), blocks.end(), int64_t{0},
                    [](const int64_t sum, const torch::Tensor &idx) { return sum + idx.size(0); });

                for (const auto &idx: blocks)
                    backward_critic(
                        select(flat_vision, idx), select(flat_proprioception, idx),
                        select(flat_returns, idx),
                        static_cast<float>(idx.size(0)) / static_cast<float>(nb_batch_rows));

                torch::nn::utils::clip_grad_norm_(critic->parameters(), grad_norm_max);
                critic_optim->step();
            }

        set_train(false);
    }

    torch::Tensor PpgTrainer::backward_actor(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions,
        const torch::Tensor &fire_masks, const torch::Tensor &old_log_probs,
        const torch::Tensor &advantages, const float batch_fraction) const {
        const auto [mu, sigma, discrete_proba] = actor->act(vision, proprioception);
        // the fire action is a decision only when it would shoot: masked out of the
        // log-probabilities (as at collection) and of the entropy
        const auto discrete_mask = discrete_action_mask(fire_masks, discrete_proba.size(-1));

        const auto curr_continuous_log_probs =
            truncated_normal_log_pdf(continuous_actions, mu, sigma).sum(-1, true);

        const auto curr_discrete_log_probs =
            (bernoulli_log_proba(discrete_actions, discrete_proba) * discrete_mask).sum(-1, true);

        const auto log_ratio = torch::clamp(
            curr_continuous_log_probs + curr_discrete_log_probs - old_log_probs, -LOG_RATIO_MAX_ABS,
            LOG_RATIO_MAX_ABS);

        const auto ratio = torch::exp(log_ratio);

        const auto continuous_entropy = truncated_normal_entropy(mu, sigma);
        const auto discrete_entropy = bernoulli_entropy(discrete_proba) * discrete_mask;

        const auto entropy_bonus = continuous_entropy_factor * continuous_entropy.sum(-1)
                                   + discrete_entropy_factor * discrete_entropy.sum(-1);

        const auto clipped_ratio = torch::clamp(ratio, 1.f - clip_epsilon, 1.f + clip_epsilon);
        const auto surrogate = torch::min(ratio * advantages, clipped_ratio * advantages);

        const auto actor_loss = -torch::mean(surrogate + entropy_bonus);

        // weighted so that the accumulated gradient is the one of the whole rollout's mean
        (actor_loss * batch_fraction).backward();

        // metrics
        actor_mean_loss_metric->add(actor_loss.cpu().item<float>());
        continuous_entropy_metric->add(continuous_entropy.mean().item<float>());
        // mean over the discrete actions that had an effect
        discrete_entropy_metric->add(
            (discrete_entropy.sum() / discrete_mask.sum().clamp_min(1.f)).item<float>());

        clip_fraction_metric->add(
            ((ratio - 1.f).abs() > clip_epsilon).to(torch::kFloat).mean().item<float>());

        return (ratio - 1.f - log_ratio).detach().flatten();
    }

    void PpgTrainer::backward_actor_aux(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &fire_masks, const torch::Tensor &target_values,
        const torch::Tensor &old_continuous_actions, const torch::Tensor &old_continuous_log_probs,
        const torch::Tensor &old_discrete_probas, const float batch_fraction) const {
        const auto values = actor->value(vision, proprioception);
        const auto [mu, sigma, discrete_proba] = actor->act(vision, proprioception);

        const auto aux_loss = torch::mse_loss(values, target_values, at::Reduction::None);

        // KL(old || new): sampled for the truncated normal, closed form for the Bernoulli
        const auto continuous_kl =
            old_continuous_log_probs
            - truncated_normal_log_pdf(old_continuous_actions, mu, sigma).sum(-1, true);

        const auto old_p = old_discrete_probas.clamp(EPSILON, 1.f - EPSILON);
        const auto new_p = discrete_proba.clamp(EPSILON, 1.f - EPSILON);
        // a masked fire is not a decision: kept out of the clone term as of the policy phase
        const auto discrete_kl = ((old_p * torch::log(old_p / new_p)
                                   + (1.f - old_p) * torch::log((1.f - old_p) / (1.f - new_p)))
                                  * discrete_action_mask(fire_masks, discrete_proba.size(-1)))
                                     .sum(-1, true);

        const auto loss = torch::mean(aux_loss + beta_clone * (continuous_kl + discrete_kl));

        // weighted so that the accumulated gradient is the one of the whole batch's mean
        (loss * batch_fraction).backward();
    }

    void PpgTrainer::backward_critic(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &returns, const float batch_fraction) const {
        const auto values = critic->value(vision, proprioception);
        const auto critic_loss = torch::mse_loss(values, returns, at::Reduction::Mean);

        (critic_loss * batch_fraction).backward();

        critic_mean_loss_metric->add(critic_loss.cpu().item<float>());

        const auto residual_var = (returns - values.detach()).var(false);
        const auto returns_var = returns.var(false);
        explained_variance_metric->add(
            (1.f - residual_var / returns_var.clamp_min(EPSILON)).item<float>());
    }

    PpgGaeResult
    PpgTrainer::compute_gae(const PpgRollout &rollout, const torch::Device device) const {
        torch::NoGradGuard no_grad;

        const auto nb_steps = rollout.rewards.size(0);
        const auto nb_tanks = rollout.rewards.size(1);

        // critic forward in device-sized chunks: the whole rollout's activations do not fit
        const auto eval_values = [&](const torch::Tensor &vision,
                                     const torch::Tensor &proprioception) {
            std::vector<torch::Tensor> chunks;
            for (int64_t i = 0; i < vision.size(0); i += critic_gpu_batch_size) {
                const auto end = std::min<int64_t>(i + critic_gpu_batch_size, vision.size(0));
                chunks.push_back(critic
                                     ->value(
                                         vision.slice(0, i, end).to(device),
                                         proprioception.slice(0, i, end).to(device))
                                     .cpu());
            }
            return torch::cat(chunks, 0);
        };

        const auto values =
            eval_values(
                flatten_steps(rollout.states.vision), flatten_steps(rollout.states.proprioception))
                .reshape({nb_steps, nb_tanks, 1});

        // next values are the values shifted by one step, closed by the bootstrap state
        const auto bootstrap_value =
            eval_values(rollout.bootstrap_state.vision, rollout.bootstrap_state.proprioception)
                .unsqueeze(0);
        const auto next_values = torch::cat({values.slice(0, 1), bootstrap_value}, 0);

        const auto rewards = return_scaler->scale(rollout.rewards, rollout.dones, rollout.valids);
        const auto dones = rollout.dones.to(torch::kFloat);
        const auto truncateds = rollout.truncateds.to(torch::kFloat);
        const auto valids = rollout.valids.to(torch::kFloat);

        // a truncated step bootstraps on its own value (the last alive observation):
        // the post-mortem next state is not a state the counterfactual life would reach
        const auto deltas =
            rewards + gamma * (next_values * (1.f - dones) + values * truncateds) - values;

        auto advantages = torch::zeros_like(deltas);
        auto gae = torch::zeros({nb_tanks, 1}, deltas.options());
        for (int64_t t = nb_steps - 1; t >= 0; t--) {
            gae = deltas[t] + gamma * gae_lambda * (1.f - dones[t]) * gae;
            advantages[t] = gae;
        }

        const auto returns = advantages + values;

        const auto nb_valid = valids.sum().clamp_min(1.f);
        const auto advantage_mean = torch::sum(advantages * valids) / nb_valid;
        const auto advantage_std =
            torch::sqrt(torch::sum(torch::square(advantages - advantage_mean) * valids) / nb_valid);
        advantages = (advantages - advantage_mean) / (advantage_std + EPSILON);

        return {.advantages = advantages, .returns = returns};
    }

    std::vector<std::shared_ptr<AbstractMetric>> PpgTrainer::get_metrics() {
        return {
            actor_mean_loss_metric,
            critic_mean_loss_metric,
            explained_variance_metric,
            continuous_entropy_metric,
            discrete_entropy_metric,
            clip_fraction_metric,
            kl_metric,
            skip_fraction_metric};
    }

    void PpgTrainer::save(const std::filesystem::path &output_folder) {
        // Models
        save_torch(output_folder, actor, "actor.pt");
        save_torch(output_folder, critic, "critic.pt");

        // Optimizers
        save_torch(output_folder, actor_optim, "actor_optim.pt");
        save_torch(output_folder, critic_optim, "critic_optim.pt");
        save_torch(output_folder, return_scaler, "return_scaler.pt");

        // string repr
        std::ostringstream actor_repr_oss;
        dump_module_tree(actor, actor_repr_oss, 0, "actor");
        std::ofstream actor_repr_file(output_folder / "actor_repr.txt");
        actor_repr_file << actor_repr_oss.str();
        actor_repr_file.close();

        std::ostringstream critic_repr_oss;
        dump_module_tree(critic, critic_repr_oss, 0, "critic");
        std::ofstream critic_repr_file(output_folder / "critic_repr.txt");
        critic_repr_file << critic_repr_oss.str();
        critic_repr_file.close();
    }

    void PpgTrainer::set_train(const bool train) const {
        actor->train(train);
        critic->train(train);
    }

    void PpgTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);
    }

    int PpgTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters());
    }
}// namespace arenai::agent
