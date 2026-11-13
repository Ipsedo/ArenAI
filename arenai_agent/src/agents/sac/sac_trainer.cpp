#include "./sac_trainer.h"

#include <algorithm>
#include <fstream>

#include "../../distributions/bernoulli.h"
#include "../../distributions/gaussian_tanh.h"
#include "../../metrics/mean_metric.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/target_update.h"
#include "../../networks_utils/torch_saver.h"
#include "./sac_discrete.h"

namespace arenai::agent {

    namespace {
        SacBatch slice_batch(const SacBatch &batch, const int64_t start, const int64_t end) {
            return {
                .states = batch.states.slice(start, end),
                .actions =
                    {.continuous_action = batch.actions.continuous_action.slice(0, start, end),
                     .discrete_action = batch.actions.discrete_action.slice(0, start, end)},
                .rewards = batch.rewards.slice(0, start, end),
                .bootstrap_discounts = batch.bootstrap_discounts.slice(0, start, end),
                .next_states = batch.next_states.slice(start, end)};
        }

        void write_module_repr(
            const std::filesystem::path &output_folder,
            const std::shared_ptr<torch::nn::Module> &module, const std::string &name) {
            std::ostringstream oss;
            dump_module_tree(module, oss, 0, name);
            std::ofstream file(output_folder / (name + "_repr.txt"));
            file << oss.str();
        }
    }// namespace

    SacTrainer::SacTrainer(
        const std::shared_ptr<SacActor> &actor, const std::shared_ptr<StateEncoder> &encoder,
        const std::shared_ptr<SacReplayBuffer> &replay_buffer, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions,
        const float actor_learning_rate, const float critic_learning_rate,
        const float alpha_learning_rate, const int hidden_size_sensors,
        const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float tau, const float initial_alpha,
        const float target_sigma, const float discrete_target_entropy_ratio,
        const int learning_starts, const int gradient_steps, const int batch_size,
        const int gpu_batch_size)
        : actor(actor), replay_buffer(replay_buffer),
          critic(std::make_shared<TwinQNetwork>(
              encoder, nb_continuous_actions, nb_discrete_actions, critic_hidden_sizes)),
          target_critic(std::make_shared<TwinQNetwork>(
              std::make_shared<StateEncoder>(
                  nb_sensors, hidden_size_sensors, vision_channels, group_norm_nums),
              nb_continuous_actions, nb_discrete_actions, critic_hidden_sizes)),
          alphas(std::make_shared<SacAlpha>(
              initial_alpha, nb_continuous_actions + nb_discrete_actions)),
          actor_optim(std::make_unique<torch::optim::Adam>(
              this->actor->head_parameters(), actor_learning_rate)),
          critic_optim(
              std::make_unique<torch::optim::Adam>(critic->parameters(), critic_learning_rate)),
          alpha_optim(
              std::make_unique<torch::optim::Adam>(alphas->parameters(), alpha_learning_rate)),
          critic_loss_metric(std::make_shared<MeanMetric>("q", metric_window_size)),
          actor_loss_metric(std::make_shared<MeanMetric>("π", metric_window_size)),
          q_value_metric(std::make_shared<MeanMetric>("Q", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          nb_continuous_actions(nb_continuous_actions), tau(tau),
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

        struct Block {
            int64_t start;
            int64_t end;
            float fraction;
        };
        std::vector<Block> blocks;
        for (int64_t start = 0; start < batch_size; start += gpu_batch_size) {
            const auto end = std::min<int64_t>(start + gpu_batch_size, batch_size);
            blocks.push_back(
                {start, end, static_cast<float>(end - start) / static_cast<float>(batch_size)});
        }

        // critics on the frozen target
        critic_optim->zero_grad();
        for (const auto &[start, end, fraction]: blocks)
            backward_critic(slice_batch(batch, start, end), fraction);
        critic_optim->step();

        // actor through the fresh critics, temperatures on the actor's entropies
        actor_optim->zero_grad();
        alpha_optim->zero_grad();
        for (const auto &[start, end, fraction]: blocks)
            backward_actor_and_alphas(slice_batch(batch, start, end), fraction);
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

            const auto [mu, sigma, discrete_proba] = actor->act_from_encoding(
                critic->encode(block.next_states.get_vision(), block.next_states.get_state()));
            const auto [next_action, u] = gaussian_tanh_sample(mu, sigma);
            // each axis' log-probability weighted by its own temperature
            const auto continuous_log_prob = gaussian_tanh_log_pdf(u, mu, sigma);
            const auto continuous_bonus = (continuous_alpha * continuous_log_prob).sum(-1, true);

            const auto joint = discrete_joint(
                discrete_proba, TorchState::discrete_action_mask(
                                    block.next_states.get_fire_mask(), nb_discrete_actions));
            const auto discrete_bonus = (discrete_alpha * joint.action_log_probas).sum(-1);

            const auto [q1, q2] = target_critic->q(
                block.next_states.get_vision(), block.next_states.get_state(), next_action);
            const auto q_min = torch::min(q1, q2);

            // soft value: exact over the discrete combinations, one sample on the continuous
            const auto next_value =
                (joint.probas * (q_min - discrete_bonus)).sum(-1, true) - continuous_bonus;

            target = block.rewards + block.bootstrap_discounts * next_value;
        }

        const auto [q1, q2] = critic->q(
            block.states.get_vision(), block.states.get_state(), block.actions.continuous_action);
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

        const auto encoding =
            critic->encode(block.states.get_vision(), block.states.get_state()).detach();
        const auto [mu, sigma, discrete_proba] = actor->act_from_encoding(encoding);
        const auto [action, u] = gaussian_tanh_sample(mu, sigma);
        // [N, nb_continuous_actions]
        const auto continuous_log_prob = gaussian_tanh_log_pdf(u, mu, sigma);

        const auto discrete_mask =
            TorchState::discrete_action_mask(block.states.get_fire_mask(), nb_discrete_actions);
        const auto joint = discrete_joint(discrete_proba, discrete_mask);

        const auto [q1, q2] = critic->q_from_encoding(encoding, action);
        const auto q_min = torch::min(q1, q2);

        const auto discrete_bonus = (discrete_alpha * joint.action_log_probas).sum(-1);
        const auto actor_loss = torch::mean(
            (continuous_alpha * continuous_log_prob).sum(-1, true)
            + (joint.probas * (discrete_bonus - q_min)).sum(-1, true));
        (actor_loss * batch_fraction).backward();

        const auto discrete_entropies = discrete_action_entropies(joint).detach();
        backward_alphas(continuous_log_prob, discrete_entropies, discrete_mask, batch_fraction);

        actor_loss_metric->add(actor_loss.item<float>());
        continuous_entropy_metric->add(-continuous_log_prob.sum(-1).mean().item<float>());
        discrete_entropy_metric->add(discrete_entropies.sum(-1).mean().item<float>());
    }

    void SacTrainer::backward_alphas(
        const torch::Tensor &continuous_log_prob, const torch::Tensor &discrete_entropies,
        const torch::Tensor &discrete_mask, const float batch_fraction) const {
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

        const auto alpha = alphas->alpha().detach().cpu();
        for (size_t i = 0; i < alpha_metrics.size(); i++)
            alpha_metrics[i]->add(alpha[static_cast<int64_t>(i)].item<float>());
    }

    std::vector<std::shared_ptr<AbstractMetric>> SacTrainer::get_metrics() {
        return {
            critic_loss_metric, actor_loss_metric, q_value_metric, continuous_entropy_metric,
            discrete_entropy_metric};
    }

    std::vector<std::shared_ptr<AbstractMetric>> SacTrainer::get_csv_only_metrics() {
        return {alpha_metrics.begin(), alpha_metrics.end()};
    }

    void SacTrainer::save(const std::filesystem::path &output_folder) {
        save_torch(output_folder, actor, "actor.pt");
        save_torch(output_folder, critic, "critic.pt");
        save_torch(output_folder, target_critic, "target_critic.pt");
        save_torch(output_folder, alphas, "alphas.pt");

        save_torch(output_folder, actor_optim, "actor_optim.pt");
        save_torch(output_folder, critic_optim, "critic_optim.pt");
        save_torch(output_folder, alpha_optim, "alpha_optim.pt");

        write_module_repr(output_folder, actor, "actor");
        write_module_repr(output_folder, critic, "critic");
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
        return count_parameters_impl(actor->head_parameters())
               + count_parameters_impl(critic->parameters())
               + count_parameters_impl(alphas->parameters());
    }

}// namespace arenai::agent
