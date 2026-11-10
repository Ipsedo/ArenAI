//
// Created by claude on 08/11/2026.
//

#include "./sac_trainer.h"

#include <algorithm>
#include <fstream>

#include "../../distributions/bernoulli.h"
#include "../../distributions/gaussian_tanh.h"
#include "../../metrics/mean_metric.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/target_update.h"
#include "../../networks_utils/torch_converter.h"
#include "../../networks_utils/torch_saver.h"
#include "./sac_discrete.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        TorchState slice_state(const TorchState &state, const int64_t start, const int64_t end) {
            return {
                .vision = state.vision.slice(0, start, end),
                .proprioception = state.proprioception.slice(0, start, end),
                .fire_mask = state.fire_mask.slice(0, start, end)};
        }

        SacBatch slice_batch(const SacBatch &batch, const int64_t start, const int64_t end) {
            return {
                .states = slice_state(batch.states, start, end),
                .actions =
                    {.continuous_action = batch.actions.continuous_action.slice(0, start, end),
                     .discrete_action = batch.actions.discrete_action.slice(0, start, end)},
                .rewards = batch.rewards.slice(0, start, end),
                .cuts = batch.cuts.slice(0, start, end),
                .next_states = slice_state(batch.next_states, start, end)};
        }
    }// namespace

    SacTrainer::SacTrainer(
        const std::shared_ptr<SacActor> &actor,
        const std::shared_ptr<SacReplayBuffer> &replay_buffer, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions,
        const float actor_learning_rate, const float critic_learning_rate,
        const float alpha_learning_rate, const int hidden_size_sensors,
        const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float gamma, const float tau, const float initial_alpha,
        const float target_sigma, const float discrete_target_entropy_ratio,
        const int learning_starts, const int gradient_steps, const int batch_size,
        const int gpu_batch_size)
        : actor(actor), replay_buffer(replay_buffer),
          critic(std::make_shared<TwinQNetwork>(
              nb_sensors, nb_continuous_actions, nb_discrete_actions, hidden_size_sensors,
              critic_hidden_sizes, vision_channels, group_norm_nums)),
          target_critic(std::make_shared<TwinQNetwork>(
              nb_sensors, nb_continuous_actions, nb_discrete_actions, hidden_size_sensors,
              critic_hidden_sizes, vision_channels, group_norm_nums)),
          alphas(std::make_shared<SacAlpha>(
              initial_alpha, nb_continuous_actions + nb_discrete_actions)),
          actor_optim(
              std::make_unique<torch::optim::Adam>(this->actor->parameters(), actor_learning_rate)),
          critic_optim(
              std::make_unique<torch::optim::Adam>(critic->parameters(), critic_learning_rate)),
          alpha_optim(
              std::make_unique<torch::optim::Adam>(alphas->parameters(), alpha_learning_rate)),
          critic_loss_metric(std::make_shared<MeanMetric>("q", metric_window_size)),
          actor_loss_metric(std::make_shared<MeanMetric>("π", metric_window_size)),
          q_value_metric(std::make_shared<MeanMetric>("Q", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          nb_continuous_actions(nb_continuous_actions), gamma(gamma), tau(tau),
          continuous_target_entropy(gaussian_tanh_target_entropy(1, target_sigma)),
          discrete_target_entropy_ratio(discrete_target_entropy_ratio),
          learning_starts(learning_starts), gradient_steps(gradient_steps), batch_size(batch_size),
          gpu_batch_size(gpu_batch_size) {
        for (int i = 0; i < nb_continuous_actions; i++)
            alpha_metrics.push_back(std::make_shared<MeanMetric>(
                "αc" + std::to_string(i), metric_window_size, 2, true));
        for (int i = 0; i < nb_discrete_actions; i++)
            alpha_metrics.push_back(std::make_shared<MeanMetric>(
                "αd" + std::to_string(i), metric_window_size, 2, true));

        hard_update(target_critic, critic);
        for (auto &parameter: target_critic->parameters()) parameter.set_requires_grad(false);

        to(device);
        set_train(false);
    }

    void SacTrainer::step() {
        if (replay_buffer->nb_transitions() >= static_cast<int64_t>(learning_starts)) train();
    }

    void SacTrainer::train() const {
        set_train(true);
        for (int g = 0; g < gradient_steps; g++) train_step();
        set_train(false);
    }

    void SacTrainer::train_step() const {
        const auto device = actor->parameters().back().device();
        const auto batch = replay_buffer->sample(batch_size, device);

        std::vector<std::pair<int64_t, int64_t>> blocks;
        for (int64_t start = 0; start < batch_size; start += gpu_batch_size)
            blocks.emplace_back(start, std::min<int64_t>(start + gpu_batch_size, batch_size));

        const auto fraction = [&](const std::pair<int64_t, int64_t> &block) {
            return static_cast<float>(block.second - block.first) / static_cast<float>(batch_size);
        };

        // critics on the frozen target
        critic_optim->zero_grad();
        for (const auto &block: blocks)
            backward_critic(slice_batch(batch, block.first, block.second), fraction(block));
        critic_optim->step();

        // actor through the fresh critics, temperatures on the actor's entropies
        actor_optim->zero_grad();
        alpha_optim->zero_grad();
        for (const auto &block: blocks)
            backward_actor_and_alphas(
                slice_batch(batch, block.first, block.second), fraction(block));
        actor_optim->step();
        alpha_optim->step();

        soft_update(target_critic, critic, tau);
    }

    std::pair<torch::Tensor, torch::Tensor> SacTrainer::split_alphas() const {
        const auto alpha = alphas->alpha().detach();
        return {alpha.slice(0, 0, nb_continuous_actions), alpha.slice(0, nb_continuous_actions)};
    }

    void SacTrainer::backward_critic(const SacBatch &block, const float batch_fraction) const {
        const auto [continuous_alpha, discrete_alpha] = split_alphas();
        const auto nb_discrete_actions = block.actions.discrete_action.size(-1);

        torch::Tensor target;
        {
            torch::NoGradGuard no_grad;

            const auto [mu, sigma, discrete_proba] =
                actor->act(block.next_states.vision, block.next_states.proprioception);
            const auto [next_action, u] = gaussian_tanh_sample(mu, sigma);
            // each axis' log-probability weighted by its own temperature
            const auto continuous_log_prob = gaussian_tanh_log_pdf(u, mu, sigma);
            const auto continuous_bonus = (continuous_alpha * continuous_log_prob).sum(-1, true);

            const auto joint = discrete_joint(
                discrete_proba,
                discrete_action_mask(block.next_states.fire_mask, nb_discrete_actions));
            const auto discrete_bonus = (discrete_alpha * joint.action_log_probas).sum(-1);

            const auto [q1, q2] = target_critic->q(
                block.next_states.vision, block.next_states.proprioception, next_action);
            const auto q_min = torch::min(q1, q2);

            // soft value: exact over the discrete combinations, one sample on the continuous
            const auto next_value =
                (joint.probas * (q_min - discrete_bonus)).sum(-1, true) - continuous_bonus;

            target = block.rewards + gamma * (1.f - block.cuts) * next_value;
        }

        const auto [q1, q2] = critic->q(
            block.states.vision, block.states.proprioception, block.actions.continuous_action);
        const auto taken = combination_index(block.actions.discrete_action).unsqueeze(-1);
        const auto q1_taken = q1.gather(1, taken);
        const auto q2_taken = q2.gather(1, taken);

        const auto critic_loss =
            torch::mse_loss(q1_taken, target) + torch::mse_loss(q2_taken, target);
        (critic_loss * batch_fraction).backward();

        critic_loss_metric->add(critic_loss.item<float>());
        q_value_metric->add(q1_taken.mean().item<float>());
    }

    void
    SacTrainer::backward_actor_and_alphas(const SacBatch &block, const float batch_fraction) const {
        const auto [continuous_alpha, discrete_alpha] = split_alphas();
        const auto nb_discrete_actions = block.actions.discrete_action.size(-1);

        const auto [mu, sigma, discrete_proba] =
            actor->act(block.states.vision, block.states.proprioception);
        const auto [action, u] = gaussian_tanh_sample(mu, sigma);
        // [N, nb_continuous_actions]
        const auto continuous_log_prob = gaussian_tanh_log_pdf(u, mu, sigma);

        const auto discrete_mask =
            discrete_action_mask(block.states.fire_mask, nb_discrete_actions);
        const auto joint = discrete_joint(discrete_proba, discrete_mask);

        const auto [q1, q2] = critic->q(block.states.vision, block.states.proprioception, action);
        const auto q_min = torch::min(q1, q2);

        const auto discrete_bonus = (discrete_alpha * joint.action_log_probas).sum(-1);
        const auto actor_loss = torch::mean(
            (continuous_alpha * continuous_log_prob).sum(-1, true)
            + (joint.probas * (discrete_bonus - q_min)).sum(-1, true));
        (actor_loss * batch_fraction).backward();

        // temperatures: one per action, each pushed toward its own entropy target
        // (Lagrangian on log-alpha); a discrete action without effect has a zero target
        const auto discrete_entropies = discrete_action_entropies(joint).detach();
        const auto discrete_target_entropies =
            discrete_target_entropy_ratio * bernoulli_maximum_entropy() * discrete_mask;

        const auto log_alpha = torch::log(alphas->alpha());
        const auto continuous_log_alpha = log_alpha.slice(0, 0, nb_continuous_actions);
        const auto discrete_log_alpha = log_alpha.slice(0, nb_continuous_actions);
        const auto alpha_loss =
            -(continuous_log_alpha
              * (continuous_log_prob.detach() + continuous_target_entropy).mean(0))
                 .sum()
            - (discrete_log_alpha * (discrete_target_entropies - discrete_entropies).mean(0)).sum();
        (alpha_loss * batch_fraction).backward();

        actor_loss_metric->add(actor_loss.item<float>());
        continuous_entropy_metric->add(-continuous_log_prob.sum(-1).mean().item<float>());
        discrete_entropy_metric->add(discrete_entropies.sum(-1).mean().item<float>());
        const auto alpha = alphas->alpha().detach().cpu();
        for (size_t i = 0; i < alpha_metrics.size(); i++)
            alpha_metrics[i]->add(alpha[static_cast<int64_t>(i)].item<float>());
    }

    std::vector<std::shared_ptr<AbstractMetric>> SacTrainer::get_metrics() {
        std::vector<std::shared_ptr<AbstractMetric>> metrics{
            critic_loss_metric, actor_loss_metric, q_value_metric, continuous_entropy_metric,
            discrete_entropy_metric};
        metrics.insert(metrics.end(), alpha_metrics.begin(), alpha_metrics.end());
        return metrics;
    }

    void SacTrainer::save(const std::filesystem::path &output_folder) {
        save_torch(output_folder, actor, "actor.pt");
        save_torch(output_folder, critic, "critic.pt");
        save_torch(output_folder, target_critic, "target_critic.pt");
        save_torch(output_folder, alphas, "alphas.pt");

        save_torch(output_folder, actor_optim, "actor_optim.pt");
        save_torch(output_folder, critic_optim, "critic_optim.pt");
        save_torch(output_folder, alpha_optim, "alpha_optim.pt");

        std::ostringstream actor_repr_oss;
        dump_module_tree(actor, actor_repr_oss, 0, "actor");
        std::ofstream actor_repr_file(output_folder / "actor_repr.txt");
        actor_repr_file << actor_repr_oss.str();

        std::ostringstream critic_repr_oss;
        dump_module_tree(critic, critic_repr_oss, 0, "critic");
        std::ofstream critic_repr_file(output_folder / "critic_repr.txt");
        critic_repr_file << critic_repr_oss.str();
    }

    void SacTrainer::set_train(const bool train) const {
        actor->train(train);
        critic->train(train);
        target_critic->train(false);
    }

    void SacTrainer::to(const torch::Device device) const {
        actor->to(device);
        critic->to(device);
        target_critic->to(device);
        alphas->to(device);
    }

    int SacTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic->parameters())
               + count_parameters_impl(alphas->parameters());
    }

}// namespace arenai::agent
