//
// Created by claude on 26/10/2026.
//

#include "./impala_ppo_trainer.h"

#include <algorithm>
#include <fstream>

#include "../../distributions/bernoulli.h"
#include "../../distributions/truncated_normal.h"
#include "../../metrics/mean_metric.h"
#include "../../networks/constants.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/torch_converter.h"
#include "../../networks_utils/torch_saver.h"
#include "../ppo_update.h"

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

    ImpalaPpoTrainer::ImpalaPpoTrainer(
        const std::shared_ptr<ImpalaActor> &actor,
        const std::shared_ptr<PpoRolloutBuffer> &rollout_buffer, const int nb_sensors,
        const float actor_learning_rate, const float critic_learning_rate,
        const int hidden_size_sensors, const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels, const torch::Device device,
        const int metric_window_size, const float gamma, const float gae_lambda,
        const float clip_epsilon, const float target_kl, const float grad_norm_max,
        const float continuous_entropy_factor, const float discrete_entropy_factor,
        const int epochs, const int rollout_size, const int minibatch_size)
        : actor(actor), rollout_buffer(rollout_buffer),
          continuous_entropy_factor(continuous_entropy_factor),
          discrete_entropy_factor(discrete_entropy_factor),
          critic(std::make_shared<ImpalaValueFunction>(
              nb_sensors, hidden_size_sensors, critic_hidden_sizes, vision_channels)),
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
          grad_norm_max(grad_norm_max), epochs(epochs), rollout_size(rollout_size),
          minibatch_size(minibatch_size) {
        to(device);

        set_train(false);
    }

    void ImpalaPpoTrainer::step() {
        if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(rollout_size)) train();
    }

    void ImpalaPpoTrainer::train() const {
        const auto device = actor->parameters().back().device();

        const auto rollout = rollout_buffer->get_rollout();

        set_train(false);
        const auto [advantages, returns] = compute_gae(rollout, device);

        set_train(true);

        // the rollout stays on CPU; only the minibatches hit the device
        const auto flat_vision = flatten_steps(rollout.states.vision);
        const auto flat_proprioception = flatten_steps(rollout.states.proprioception);
        const auto flat_continuous_actions = flatten_steps(rollout.actions.continuous_action);
        const auto flat_discrete_actions = flatten_steps(rollout.actions.discrete_action);
        const auto flat_fire_masks = flatten_steps(rollout.states.fire_mask);
        const auto flat_old_log_probs =
            flatten_steps(rollout.continuous_log_probs) + flatten_steps(rollout.discrete_log_probs);
        const auto flat_advantages = flatten_steps(advantages);
        const auto flat_returns = flatten_steps(returns);

        // live (step, tank) pairs; the minibatches split these rows only
        const auto valid_idx =
            torch::nonzero(flatten_steps(rollout.valids).squeeze(-1)).squeeze(-1);
        const auto nb_valid_rows = valid_idx.size(0);
        if (nb_valid_rows == 0) return;

        const auto select = [&](const torch::Tensor &tensor, const torch::Tensor &idx) {
            return tensor.index_select(0, idx).to(device);
        };

        // one update per epoch: the minibatches' gradients are accumulated over the whole
        // rollout, minibatch_size only bounds the device memory of a forward / backward pass
        for (int e = 0; e < epochs; e++) {
            actor_optim->zero_grad();
            critic_optim->zero_grad();

            std::vector<torch::Tensor> kl_per_rows;

            for (int64_t start = 0; start < nb_valid_rows; start += minibatch_size) {
                const auto idx = valid_idx.slice(
                    0, start, std::min<int64_t>(start + minibatch_size, nb_valid_rows));
                const auto rollout_fraction =
                    static_cast<float>(idx.size(0)) / static_cast<float>(nb_valid_rows);

                const auto mb_vision = select(flat_vision, idx);
                const auto mb_proprioception = select(flat_proprioception, idx);

                kl_per_rows.push_back(backward_actor(
                    mb_vision, mb_proprioception, select(flat_continuous_actions, idx),
                    select(flat_discrete_actions, idx), select(flat_fire_masks, idx),
                    select(flat_old_log_probs, idx), select(flat_advantages, idx),
                    rollout_fraction));

                backward_critic(
                    mb_vision, mb_proprioception, select(flat_returns, idx), rollout_fraction);
            }

            // KL of the policy the epoch starts from: past the threshold, only the critic moves
            const auto approx_kl = trimmed_kl(kl_per_rows);
            const bool kl_exceeded = target_kl > 0.f && approx_kl > 1.5f * target_kl;

            if (!kl_exceeded) {
                torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
                actor_optim->step();
            }

            torch::nn::utils::clip_grad_norm_(critic->parameters(), grad_norm_max);
            critic_optim->step();

            kl_metric->add(approx_kl);
            skip_fraction_metric->add(kl_exceeded ? 1.f : 0.f);
        }

        set_train(false);
    }

    torch::Tensor ImpalaPpoTrainer::backward_actor(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions,
        const torch::Tensor &fire_masks, const torch::Tensor &old_log_probs,
        const torch::Tensor &advantages, const float rollout_fraction) const {
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
        (actor_loss * rollout_fraction).backward();

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

    void ImpalaPpoTrainer::backward_critic(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &returns, const float rollout_fraction) const {
        const auto values = critic->value(vision, proprioception);
        const auto critic_loss = torch::mse_loss(values, returns, at::Reduction::Mean);

        (critic_loss * rollout_fraction).backward();

        critic_mean_loss_metric->add(critic_loss.cpu().item<float>());

        const auto residual_var = (returns - values.detach()).var(false);
        const auto returns_var = returns.var(false);
        explained_variance_metric->add(
            (1.f - residual_var / returns_var.clamp_min(EPSILON)).item<float>());
    }

    ImpalaGaeResult
    ImpalaPpoTrainer::compute_gae(const PpoRollout &rollout, const torch::Device device) const {
        torch::NoGradGuard no_grad;

        const auto nb_steps = rollout.rewards.size(0);
        const auto nb_tanks = rollout.rewards.size(1);

        // critic forward in minibatch-sized chunks: the full rollout does not fit on the device
        const auto eval_values = [&](const torch::Tensor &vision,
                                     const torch::Tensor &proprioception) {
            std::vector<torch::Tensor> chunks;
            for (int64_t i = 0; i < vision.size(0); i += minibatch_size) {
                const auto end = std::min<int64_t>(i + minibatch_size, vision.size(0));
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

    std::vector<std::shared_ptr<AbstractMetric>> ImpalaPpoTrainer::get_metrics() {
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

    void ImpalaPpoTrainer::save(const std::filesystem::path &output_folder) {
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

    void ImpalaPpoTrainer::set_train(const bool train) const {
        actor->train(train);
        critic->train(train);
    }

    void ImpalaPpoTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);
    }

    int ImpalaPpoTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters());
    }
}// namespace arenai::agent
