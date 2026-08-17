//
// Created by claude on 22/07/2026.
//

#include "./ppo_trainer.h"

#include <fstream>

#include "../../distributions/multinomial.h"
#include "../../distributions/truncated_normal.h"
#include "../../metrics/mean_metric.h"
#include "../../metrics/std_metric.h"
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
    }// namespace

    PpoTrainer::PpoTrainer(
        std::shared_ptr<Actor> actor, std::shared_ptr<PpoRolloutBuffer> rollout_buffer,
        const int vision_height, const int vision_width, const int nb_sensors,
        const float actor_learning_rate, const float critic_learning_rate,
        const int hidden_size_sensors, const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float gamma, const float gae_lambda,
        const float clip_epsilon, const float target_kl, const float grad_norm_max,
        const float continuous_entropy_coefficient, const float discrete_entropy_coefficient,
        const int epochs, const int rollout_size, const int minibatch_size)
        : actor(std::move(actor)), rollout_buffer(std::move(rollout_buffer)),
          critic(std::make_shared<ValueFunction>(
              vision_height, vision_width, nb_sensors, hidden_size_sensors, critic_hidden_sizes,
              vision_channels, group_norm_nums)),
          actor_optim(std::make_unique<torch::optim::Adam>(
              this->actor->parameters(), torch::optim::AdamOptions(actor_learning_rate))),
          critic_optim(std::make_unique<torch::optim::Adam>(
              critic->parameters(), torch::optim::AdamOptions(critic_learning_rate))),
          actor_mean_loss_metric(std::make_shared<MeanMetric>("π_μ", metric_window_size)),
          actor_std_loss_metric(std::make_shared<StdMetric>("π_σ", metric_window_size)),
          critic_mean_loss_metric(std::make_shared<MeanMetric>("v_μ", metric_window_size)),
          critic_std_loss_metric(std::make_shared<StdMetric>("v_σ", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          sigma_metric(std::make_shared<MeanMetric>("σ", metric_window_size)),
          clip_fraction_metric(std::make_shared<MeanMetric>("clip", metric_window_size)),
          kl_metric(std::make_shared<MeanMetric>("kl", metric_window_size, 2, true)), gamma(gamma),
          gae_lambda(gae_lambda), clip_epsilon(clip_epsilon), target_kl(target_kl),
          grad_norm_max(grad_norm_max),
          continuous_entropy_coefficient(continuous_entropy_coefficient),
          discrete_entropy_coefficient(discrete_entropy_coefficient), epochs(epochs),
          rollout_size(rollout_size), minibatch_size(minibatch_size) {

        to(device);
    }

    void PpoTrainer::step() {
        // on-policy cadence: wait for a full rollout (rollout_size steps), consume it, start over
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

        bool kl_stop = false;
        for (int e = 0; e < epochs; e++) {
            const auto perm = valid_idx.index_select(0, torch::randperm(nb_valid_rows));

            for (int64_t start = 0; start < nb_valid_rows; start += minibatch_size) {
                const auto idx =
                    perm.slice(0, start, std::min<int64_t>(start + minibatch_size, nb_valid_rows));

                // the observation is the bulk of the transfer: uploaded once, used by both
                const auto mb_vision = select(flat_vision, idx);
                const auto mb_proprioception = select(flat_proprioception, idx);

                // actor: skipped for the rest of the update once a minibatch drifted past
                // the KL threshold
                if (!kl_stop)
                    kl_stop = train_actor(
                        mb_vision, mb_proprioception, select(flat_continuous_actions, idx),
                        select(flat_discrete_actions, idx), select(flat_old_log_probs, idx),
                        select(flat_advantages, idx));

                // critic: keeps training through the whole update, the KL early stop is a
                // policy criterion only
                train_critic(mb_vision, mb_proprioception, select(flat_returns, idx));
            }
        }
    }

    bool PpoTrainer::train_actor(
        const torch::Tensor &vision, const torch::Tensor &proprioception,
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions,
        const torch::Tensor &old_log_probs, const torch::Tensor &advantages) const {

        // policy: clipped surrogate on the joint (continuous x discrete) ratio
        const auto [mu, sigma, discrete_proba] = actor->act(vision, proprioception);

        const auto curr_continuous_log_probs =
            truncated_normal_log_pdf(continuous_actions, mu, sigma).sum(-1, true);

        const auto clamped_proba = torch::clamp(discrete_proba, EPSILON, 1.0 - EPSILON);
        const auto curr_discrete_log_probs =
            torch::sum(discrete_actions * torch::log(clamped_proba), -1, true);

        const auto ratio =
            torch::exp(curr_continuous_log_probs + curr_discrete_log_probs - old_log_probs);

        const auto continuous_entropy = truncated_normal_entropy(mu, sigma);

        // as in SAC: summed over the dimensions in the loss, averaged in the metric
        continuous_entropy_metric->add(continuous_entropy.mean().item<float>());
        sigma_metric->add(sigma.mean().item<float>());

        // already summed over the discrete actions: multinomial_entropy keeps the last dim
        const auto discrete_entropy = multinomial_entropy(discrete_proba);

        discrete_entropy_metric->add(discrete_entropy.mean().item<float>());

        // approx KL (Schulman): E[(ratio - 1) - log ratio]; skip this minibatch when the
        // policy drifted too far from the rollout policy
        const auto approx_kl = torch::mean(ratio - 1.f - torch::log(ratio)).item<float>();

        kl_metric->add(approx_kl);

        if (target_kl > 0.f && approx_kl > 1.5f * target_kl) return true;

        const auto clipped_ratio = torch::clamp(ratio, 1.f - clip_epsilon, 1.f + clip_epsilon);

        const auto surrogate = torch::min(ratio * advantages, clipped_ratio * advantages);

        // the bonuses are the only upward force on the entropies: the surrogate shrinks both
        // on its own, and neither head has a restoring force of its own
        const auto actor_loss = -torch::mean(
            surrogate + continuous_entropy_coefficient * torch::sum(continuous_entropy, -1, true)
            + discrete_entropy_coefficient * discrete_entropy);

        actor_optim->zero_grad();
        actor_loss.backward();
        torch::nn::utils::clip_grad_norm_(actor->parameters(), grad_norm_max);
        actor_optim->step();

        const auto loss_value = actor_loss.cpu().item<float>();
        actor_mean_loss_metric->add(loss_value);
        actor_std_loss_metric->add(loss_value);

        clip_fraction_metric->add(
            ((ratio - 1.f).abs() > clip_epsilon).to(torch::kFloat).mean().item<float>());

        return false;
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

        const auto loss_value = critic_loss.cpu().item<float>();
        critic_mean_loss_metric->add(loss_value);
        critic_std_loss_metric->add(loss_value);
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
        return {actor_mean_loss_metric, actor_std_loss_metric,     critic_mean_loss_metric,
                critic_std_loss_metric, continuous_entropy_metric, discrete_entropy_metric,
                sigma_metric,           clip_fraction_metric,      kl_metric};
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
    }

    void PpoTrainer::to(const torch::Device device) {
        actor->to(device);
        critic->to(device);
    }

    int PpoTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters());
    }

}// namespace arenai::agent
