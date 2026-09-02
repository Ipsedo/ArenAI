//
// Created by claude on 22/07/2026.
//

#include "./ppo_trainer.h"

#include <algorithm>
#include <fstream>

#include "../../distributions/multinomial.h"
#include "../../distributions/truncated_normal.h"
#include "../../metrics/mean_metric.h"
#include "../../networks/constants.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/torch_saver.h"

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

        constexpr float KL_TRIM_FRACTION = 0.01f;

        constexpr float CONTINUOUS_ALPHA_K_P = 2e-1f;
        constexpr float CONTINUOUS_ALPHA_K_I = 5e-3f;
        constexpr float CONTINUOUS_ALPHA_K_D = 1.f;

        constexpr float DISCRETE_ALPHA_K_P = 2e-1f;
        constexpr float DISCRETE_ALPHA_K_I = 1e-2f;
        constexpr float DISCRETE_ALPHA_K_D = 1.f;

        constexpr float ALPHA_INITIAL = 1e-3f;

    }// namespace

    PpoTrainer::PpoTrainer(
        const std::shared_ptr<Actor> &actor,
        const std::shared_ptr<PpoRolloutBuffer> &rollout_buffer, const int vision_height,
        const int vision_width, const int nb_sensors, const int nb_continuous_actions,
        const int nb_discrete_action, const float actor_learning_rate,
        const float critic_learning_rate, const int hidden_size_sensors,
        const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float gamma, const float gae_lambda,
        const float clip_epsilon, const float target_kl, const float grad_norm_max,
        const float continuous_target_entropy, const float discrete_target_entropy_factor,
        const int epochs, const int rollout_size, const int minibatch_size)
        : actor(actor), rollout_buffer(rollout_buffer),
          continuous_alpha(std::make_unique<PidLagrangianAlphaParameters>(
              CONTINUOUS_ALPHA_K_P, CONTINUOUS_ALPHA_K_I, CONTINUOUS_ALPHA_K_D, ALPHA_INITIAL,
              nb_continuous_actions)),
          discrete_alpha(std::make_unique<PidLagrangianAlphaParameters>(
              DISCRETE_ALPHA_K_P, DISCRETE_ALPHA_K_I, DISCRETE_ALPHA_K_D, ALPHA_INITIAL, 1)),
          continuous_target_entropy(continuous_target_entropy),
          discrete_target_entropy(
              discrete_target_entropy_factor * multinomial_maximum_entropy(nb_discrete_action)),
          critic(std::make_shared<ValueFunction>(
              vision_height, vision_width, nb_sensors, hidden_size_sensors, critic_hidden_sizes,
              vision_channels, group_norm_nums)),
          actor_optim(
              std::make_unique<torch::optim::Adam>(this->actor->parameters(), actor_learning_rate)),
          critic_optim(
              std::make_unique<torch::optim::Adam>(critic->parameters(), critic_learning_rate)),
          actor_mean_loss_metric(std::make_shared<MeanMetric>("π", metric_window_size)),
          critic_mean_loss_metric(std::make_shared<MeanMetric>("v", metric_window_size)),
          explained_variance_metric(std::make_shared<MeanMetric>("ev", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          continuous_alpha_metric(std::make_shared<MeanMetric>("α_c", metric_window_size, 2, true)),
          discrete_alpha_metric(std::make_shared<MeanMetric>("α_d", metric_window_size, 2, true)),
          clip_fraction_metric(std::make_shared<MeanMetric>("clip", metric_window_size)),
          kl_metric(std::make_shared<MeanMetric>("kl", metric_window_size, 2, true)),
          skip_fraction_metric(std::make_shared<MeanMetric>("skip", metric_window_size)),
          gamma(gamma), gae_lambda(gae_lambda), clip_epsilon(clip_epsilon), target_kl(target_kl),
          grad_norm_max(grad_norm_max), epochs(epochs), rollout_size(rollout_size),
          minibatch_size(minibatch_size) {
        to(device);

        set_train(false);
    }

    void PpoTrainer::step() {
        if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(rollout_size)) train();
    }

    void PpoTrainer::train() const {
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
        const auto flat_old_log_probs =
            flatten_steps(rollout.continuous_log_probs) + flatten_steps(rollout.discrete_log_probs);
        const auto flat_advantages = flatten_steps(advantages);
        const auto flat_returns = flatten_steps(returns);

        // live (step, tank) pairs; minibatches are drawn from these rows only
        const auto valid_idx =
            torch::nonzero(flatten_steps(rollout.valids).squeeze(-1)).squeeze(-1);
        const auto nb_valid_rows = valid_idx.size(0);
        if (nb_valid_rows == 0) return;

        const auto select = [&](const torch::Tensor &tensor, const torch::Tensor &idx) {
            return tensor.index_select(0, idx).to(device);
        };

        for (int e = 0; e < epochs; e++) {
            const auto perm = valid_idx.index_select(0, torch::randperm(nb_valid_rows));

            for (int64_t start = 0; start < nb_valid_rows; start += minibatch_size) {
                const auto idx =
                    perm.slice(0, start, std::min<int64_t>(start + minibatch_size, nb_valid_rows));

                const auto mb_vision = select(flat_vision, idx);
                const auto mb_proprioception = select(flat_proprioception, idx);

                train_actor(
                    mb_vision, mb_proprioception, select(flat_continuous_actions, idx),
                    select(flat_discrete_actions, idx), select(flat_old_log_probs, idx),
                    select(flat_advantages, idx));

                train_critic(mb_vision, mb_proprioception, select(flat_returns, idx));
            }
        }

        set_train(false);
    }

    bool PpoTrainer::train_actor(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions,
        const torch::Tensor &old_log_probs, const torch::Tensor &advantages) const {
        const auto device = actor->parameters().back().device();

        const auto [mu, sigma, discrete_proba] = actor->act(vision, proprioception);

        const auto curr_continuous_log_probs =
            truncated_normal_log_pdf(continuous_actions, mu, sigma).sum(-1, true);

        const auto clamped_proba = torch::clamp(discrete_proba, EPSILON, 1.0 - EPSILON);
        const auto curr_discrete_log_probs =
            torch::sum(discrete_actions * torch::log(clamped_proba), -1, true);

        const auto log_ratio = torch::clamp(
            curr_continuous_log_probs + curr_discrete_log_probs - old_log_probs, -LOG_RATIO_MAX_ABS,
            LOG_RATIO_MAX_ABS);

        const auto ratio = torch::exp(log_ratio);

        const auto continuous_entropy = truncated_normal_entropy(mu, sigma);
        const auto discrete_entropy = multinomial_entropy(discrete_proba);

        const auto kl_per_row = (ratio - 1.f - log_ratio).flatten();

        const auto nb_kept = std::max<int64_t>(
            1, static_cast<int64_t>(
                   static_cast<float>(kl_per_row.size(0)) * (1.f - KL_TRIM_FRACTION)));

        // per-row KL is non-negative: sorting ascending puts the outliers past nb_kept
        const auto approx_kl =
            std::get<0>(torch::sort(kl_per_row)).slice(0, 0, nb_kept).mean().item<float>();

        const bool kl_exceeded = target_kl > 0.f && approx_kl > 1.5f * target_kl;

        const auto entropy_bonus =
            torch::sum(continuous_alpha->alpha().detach() * continuous_entropy, -1)
            + discrete_alpha->alpha().squeeze(1).detach() * discrete_entropy;

        if (!kl_exceeded) {
            const auto clipped_ratio = torch::clamp(ratio, 1.f - clip_epsilon, 1.f + clip_epsilon);
            const auto surrogate = torch::min(ratio * advantages, clipped_ratio * advantages);

            const auto actor_loss = -torch::mean(surrogate + entropy_bonus);

            actor_optim->zero_grad();
            actor_loss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
            actor_optim->step();

            // actor metrics
            actor_mean_loss_metric->add(actor_loss.cpu().item<float>());
        }

        // adjust alphas
        continuous_alpha->update(
            continuous_entropy, torch::tensor(continuous_target_entropy, device));
        discrete_alpha->update(discrete_entropy, torch::tensor(discrete_target_entropy, device));

        // metrics
        continuous_entropy_metric->add(continuous_entropy.mean().item<float>());
        discrete_entropy_metric->add(discrete_entropy.mean().item<float>());

        continuous_alpha_metric->add(continuous_alpha->alpha().mean().item<float>());
        discrete_alpha_metric->add(discrete_alpha->alpha().mean().item<float>());

        kl_metric->add(approx_kl);

        clip_fraction_metric->add(
            ((ratio - 1.f).abs() > clip_epsilon).to(torch::kFloat).mean().item<float>());

        skip_fraction_metric->add(kl_exceeded ? 1.f : 0.f);

        return kl_exceeded;
    }

    void PpoTrainer::train_critic(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &returns) const {
        const auto values = critic->value(vision, proprioception);
        const auto critic_loss = torch::mse_loss(values, returns, at::Reduction::Mean);

        critic_optim->zero_grad();
        critic_loss.backward();
        torch::nn::utils::clip_grad_norm_(critic->parameters(), grad_norm_max);
        critic_optim->step();

        critic_mean_loss_metric->add(critic_loss.cpu().item<float>());

        const auto residual_var = (returns - values.detach()).var(false);
        const auto returns_var = returns.var(false);
        explained_variance_metric->add(
            (1.f - residual_var / returns_var.clamp_min(EPSILON)).item<float>());
    }

    GaeResult PpoTrainer::compute_gae(const PpoRollout &rollout, const torch::Device device) const {
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

        const auto rewards = rollout.rewards.to(torch::kFloat);
        const auto dones = rollout.dones.to(torch::kFloat);
        const auto valids = rollout.valids.to(torch::kFloat);

        const auto deltas = rewards + gamma * next_values * (1.f - dones) - values;

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

    std::vector<std::shared_ptr<AbstractMetric>> PpoTrainer::get_metrics() {
        return {actor_mean_loss_metric,    critic_mean_loss_metric, explained_variance_metric,
                continuous_entropy_metric, continuous_alpha_metric, discrete_entropy_metric,
                discrete_alpha_metric,     clip_fraction_metric,    kl_metric,
                skip_fraction_metric};
    }

    void PpoTrainer::save(const std::filesystem::path &output_folder) {
        // Models
        save_torch(output_folder, actor, "actor.pt");
        save_torch(output_folder, critic, "critic.pt");

        // Optimizers
        save_torch(output_folder, actor_optim, "actor_optim.pt");
        save_torch(output_folder, critic_optim, "critic_optim.pt");

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

    void PpoTrainer::set_train(const bool train) const {
        actor->train(train);
        critic->train(train);

        continuous_alpha->train(train);
        discrete_alpha->train(train);
    }

    void PpoTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);

        continuous_alpha->to(device);
        discrete_alpha->to(device);
    }

    int PpoTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters());
    }
}// namespace arenai::agent
