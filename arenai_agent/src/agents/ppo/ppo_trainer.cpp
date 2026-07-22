//
// Created by claude on 22/07/2026.
//

#include "./ppo_trainer.h"

#include <fstream>

#include <arenai_core/constants.h>

#include "../../distributions/multinomial.h"
#include "../../distributions/truncated_normal.h"
#include "../../metrics/mean_metric.h"
#include "../../metrics/std_metric.h"
#include "../../networks_io/torch_saver.h"
#include "../../networks_utils/print_module.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    PpoTrainer::PpoTrainer(
        std::shared_ptr<Actor> actor, std::shared_ptr<PpoRolloutBuffer> rollout_buffer,
        const int vision_height, const int vision_width, const int nb_sensors,
        const float actor_learning_rate, const float critic_learning_rate,
        const int hidden_size_sensors, const int hidden_size_actions,
        const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float gamma, const float gae_lambda,
        const float clip_epsilon, const float continuous_entropy_coef,
        const float discrete_entropy_coef, const int epochs, const int batch_size)
        : actor(std::move(actor)), rollout_buffer(std::move(rollout_buffer)),
          critic(std::make_shared<ValueFunction>(
              vision_height, vision_width, nb_sensors, hidden_size_sensors, hidden_size_actions,
              critic_hidden_sizes, vision_channels, group_norm_nums)),
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
          clip_fraction_metric(std::make_shared<MeanMetric>("clip", metric_window_size)),
          gamma(gamma), gae_lambda(gae_lambda), clip_epsilon(clip_epsilon),
          continuous_entropy_coef(continuous_entropy_coef),
          discrete_entropy_coef(discrete_entropy_coef), epochs(epochs), batch_size(batch_size) {

        to(device);
    }

    void PpoTrainer::step() {
        // on-policy cadence: wait for a full rollout (batch_size steps), consume it, start over
        if (rollout_buffer->nb_complete_steps() >= static_cast<size_t>(batch_size)) train();
    }

    void PpoTrainer::train() const {
        set_train(true);

        const auto device = actor->parameters().back().device();

        auto
            [states, actions, continuous_log_probs, discrete_log_probs, rewards, dones, truncateds,
             next_states, valids] = rollout_buffer->get_rollout();

        const auto nb_steps = rewards.size(0);
        const auto nb_tanks = rewards.size(1);

        const auto to_device = [&](const torch::Tensor &tensor) { return tensor.to(device); };

        const auto vision = to_device(states.vision);
        const auto proprioception = to_device(states.proprioception);
        const auto next_vision = to_device(next_states.vision);
        const auto next_proprioception = to_device(next_states.proprioception);

        const auto continuous_actions = to_device(actions.continuous_action);
        const auto discrete_actions = to_device(actions.discrete_action);

        const auto old_continuous_log_probs = to_device(continuous_log_probs);
        const auto old_discrete_log_probs = to_device(discrete_log_probs);

        const auto rewards_on_device = to_device(rewards).to(torch::kFloat);
        const auto dones_on_device = to_device(dones).to(torch::kFloat);
        const auto truncateds_on_device = to_device(truncateds).to(torch::kFloat);
        const auto valids_on_device = to_device(valids).to(torch::kFloat);

        const auto flatten = [&](const torch::Tensor &tensor) {
            auto sizes = tensor.sizes().vec();
            sizes.erase(sizes.begin());
            sizes[0] = nb_steps * nb_tanks;
            return tensor.reshape(sizes);
        };

        // GAE advantages and value targets, computed once with the pre-update networks
        torch::Tensor advantages;
        torch::Tensor returns;
        {
            torch::NoGradGuard no_grad;

            const auto values = critic->value(flatten(vision), flatten(proprioception))
                                    .reshape({nb_steps, nb_tanks, 1});
            const auto next_values =
                critic->value(flatten(next_vision), flatten(next_proprioception))
                    .reshape({nb_steps, nb_tanks, 1});

            // terminal: no bootstrap; truncated: bootstrap but stop the GAE recursion
            const auto terminals = dones_on_device * (1.f - truncateds_on_device);
            const auto boundaries = torch::max(dones_on_device, truncateds_on_device);

            const auto deltas =
                rewards_on_device + gamma * next_values * (1.f - terminals) - values;

            advantages = torch::zeros_like(deltas);
            auto gae = torch::zeros({nb_tanks, 1}, deltas.options());
            for (int64_t t = nb_steps - 1; t >= 0; t--) {
                gae = deltas[t] + gamma * gae_lambda * (1.f - boundaries[t]) * gae;
                advantages[t] = gae;
            }

            returns = advantages + values;

            const auto nb_valid = valids_on_device.sum().clamp_min(1.f);
            const auto advantage_mean = (advantages * valids_on_device).sum() / nb_valid;
            const auto advantage_std =
                (((advantages - advantage_mean).square() * valids_on_device).sum() / nb_valid)
                    .sqrt();
            advantages = (advantages - advantage_mean) / (advantage_std + core::EPSILON);
        }

        // keep only live (step, tank) pairs, then optimize on the full rollout
        const auto valid_idx = torch::nonzero(flatten(valids_on_device).squeeze(-1)).squeeze(-1);
        if (const auto nb_valid_rows = valid_idx.size(0); nb_valid_rows == 0) return;

        const auto select = [&](const torch::Tensor &tensor) {
            return flatten(tensor).index_select(0, valid_idx);
        };

        const auto valid_vision = select(vision);
        const auto valid_proprioception = select(proprioception);
        const auto valid_continuous_actions = select(continuous_actions);
        const auto valid_discrete_actions = select(discrete_actions);
        const auto valid_old_log_probs =
            select(old_continuous_log_probs) + select(old_discrete_log_probs);
        const auto valid_advantages = select(advantages);
        const auto valid_returns = select(returns);

        for (int e = 0; e < epochs; e++) {
            // policy: clipped surrogate on the joint (continuous x discrete) ratio
            const auto [mu, sigma, discrete_proba] = actor->act(valid_vision, valid_proprioception);

            const auto curr_continuous_log_probs =
                truncated_normal_log_pdf(valid_continuous_actions, mu, sigma).sum(-1, true);

            const auto clamped_proba =
                torch::clamp(discrete_proba, core::EPSILON, 1.0 - core::EPSILON);
            const auto curr_discrete_log_probs =
                (valid_discrete_actions * torch::log(clamped_proba)).sum(-1, true);

            const auto ratio = torch::exp(
                curr_continuous_log_probs + curr_discrete_log_probs - valid_old_log_probs);
            const auto clipped_ratio = torch::clamp(ratio, 1.f - clip_epsilon, 1.f + clip_epsilon);

            const auto surrogate =
                torch::min(ratio * valid_advantages, clipped_ratio * valid_advantages);

            const auto continuous_entropy = truncated_normal_entropy(mu, sigma).sum(-1, true);
            const auto discrete_entropy = multinomial_entropy(discrete_proba);

            const auto actor_loss = -torch::mean(
                surrogate + continuous_entropy_coef * continuous_entropy
                + discrete_entropy_coef * discrete_entropy);

            actor_optim->zero_grad();
            actor_loss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), GRAD_NORM_MAX);
            actor_optim->step();

            // critic
            const auto values = critic->value(valid_vision, valid_proprioception);
            const auto critic_loss = torch::mse_loss(values, valid_returns, at::Reduction::Mean);

            critic_optim->zero_grad();
            critic_loss.backward();
            torch::nn::utils::clip_grad_norm_(critic->parameters(), GRAD_NORM_MAX);
            critic_optim->step();

            // metrics
            actor_mean_loss_metric->add(actor_loss.cpu().item<float>());
            actor_std_loss_metric->add(actor_loss.cpu().item<float>());

            critic_mean_loss_metric->add(critic_loss.cpu().item<float>());
            critic_std_loss_metric->add(critic_loss.cpu().item<float>());

            continuous_entropy_metric->add(continuous_entropy.mean().item<float>());
            discrete_entropy_metric->add(discrete_entropy.mean().item<float>());

            clip_fraction_metric->add(
                ((ratio - 1.f).abs() > clip_epsilon).to(torch::kFloat).mean().item<float>());
        }
    }

    std::vector<std::shared_ptr<AbstractMetric>> PpoTrainer::get_metrics() {
        return {actor_mean_loss_metric, actor_std_loss_metric,     critic_mean_loss_metric,
                critic_std_loss_metric, continuous_entropy_metric, discrete_entropy_metric,
                clip_fraction_metric};
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

    void PpoTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);
    }

    int PpoTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters());
    }

}// namespace arenai::agent
