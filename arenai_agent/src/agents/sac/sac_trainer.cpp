//
// Created by claude on 22/07/2026.
//

#include "./sac_trainer.h"

#include <fstream>

#include "../../distributions/multinomial.h"
#include "../../distributions/truncated_normal.h"
#include "../../metrics/last_metric.h"
#include "../../metrics/mean_metric.h"
#include "../../metrics/std_metric.h"
#include "../../networks/constants.h"
#include "../../networks_utils/print_module.h"
#include "../../networks_utils/target_update.h"
#include "../../networks_utils/torch_loader.h"
#include "../../networks_utils/torch_saver.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    SacTrainer::SacTrainer(
        std::shared_ptr<Actor> actor, std::shared_ptr<SacReplayBuffer> replay_buffer,
        const int vision_height, const int vision_width, const int nb_sensors,
        const int nb_continuous_actions, const int nb_discrete_actions,
        const float actor_learning_rate, const float critic_learning_rate,
        const float alpha_learning_rate, const int hidden_size_sensors,
        const int hidden_size_actions, const std::vector<int> &critic_hidden_sizes,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, const torch::Device device,
        const int metric_window_size, const float tau, const float gamma, const int train_every,
        const int epochs, const int batch_size, const float target_sigma,
        const float target_fire_proba)
        : actor(std::move(actor)), replay_buffer(std::move(replay_buffer)),
          critic_1(std::make_shared<QFunction>(
              vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
              hidden_size_sensors, hidden_size_actions, critic_hidden_sizes, vision_channels,
              group_norm_nums)),
          critic_2(std::make_shared<QFunction>(
              vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
              hidden_size_sensors, hidden_size_actions, critic_hidden_sizes, vision_channels,
              group_norm_nums)),
          target_critic_1(std::make_shared<QFunction>(
              vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
              hidden_size_sensors, hidden_size_actions, critic_hidden_sizes, vision_channels,
              group_norm_nums)),
          target_critic_2(std::make_shared<QFunction>(
              vision_height, vision_width, nb_sensors, nb_continuous_actions, nb_discrete_actions,
              hidden_size_sensors, hidden_size_actions, critic_hidden_sizes, vision_channels,
              group_norm_nums)),
          alpha_continuous(std::make_shared<ClampedAlphaParameters>(
              0.001f, 1e-5f, 1e-2f, nb_continuous_actions)),
          alpha_discrete(std::make_shared<ClampedAlphaParameters>(0.001f, 1e-5f, 1e-2f, 1)),
          actor_optim(std::make_unique<torch::optim::Adam>(
              this->actor->parameters(), torch::optim::AdamOptions(actor_learning_rate))),
          critic_1_optim(std::make_unique<torch::optim::Adam>(
              critic_1->parameters(), torch::optim::AdamOptions(critic_learning_rate))),
          critic_2_optim(std::make_unique<torch::optim::Adam>(
              critic_2->parameters(), torch::optim::AdamOptions(critic_learning_rate))),
          alpha_continuous_optim(std::make_unique<torch::optim::Adam>(
              alpha_continuous->parameters(), torch::optim::AdamOptions(alpha_learning_rate))),
          alpha_discrete_optim(std::make_unique<torch::optim::Adam>(
              alpha_discrete->parameters(), torch::optim::AdamOptions(alpha_learning_rate))),
          actor_mean_loss_metric(std::make_shared<MeanMetric>("π_μ", metric_window_size)),
          actor_std_loss_metric(std::make_shared<StdMetric>("π_σ", metric_window_size)),
          critic_1_mean_loss_metric(std::make_shared<MeanMetric>("q1_μ", metric_window_size)),
          critic_1_std_loss_metric(std::make_shared<StdMetric>("q1_σ", metric_window_size)),
          critic_2_mean_loss_metric(std::make_shared<MeanMetric>("q2_μ", metric_window_size)),
          critic_2_std_loss_metric(std::make_shared<StdMetric>("q2_σ", metric_window_size)),
          continuous_entropy_metric(std::make_shared<MeanMetric>("Hc", metric_window_size)),
          discrete_entropy_metric(std::make_shared<MeanMetric>("Hd", metric_window_size)),
          alpha_continuous_metric(std::make_shared<MeanMetric>("α_c", metric_window_size, 2, true)),
          alpha_discrete_metric(std::make_shared<MeanMetric>("α_d", metric_window_size, 2, true)),
          continuous_target_entropy_metric(std::make_shared<LastMetric>("Hc_t")),
          discrete_target_entropy_metric(std::make_shared<LastMetric>("Hd_t")), tau(tau),
          gamma(gamma), train_every(train_every), train_counter(0), epochs(epochs),
          batch_size(batch_size),
          target_sigma(
              torch::tensor(std::vector(nb_continuous_actions, target_sigma)).unsqueeze(0)),
          target_discrete_entropy(
              torch::tensor({multinomial_target_entropy(target_fire_proba)}).unsqueeze(0)) {

        hard_update(target_critic_1, critic_1);
        hard_update(target_critic_2, critic_2);

        to(device);

        set_train(false);
    }

    void SacTrainer::step() {
        if (train_counter == train_every - 1) train();
        train_counter = (train_counter + 1) % train_every;
    }

    void SacTrainer::train() const {

        set_train(true);

        for (int e = 0; e < epochs; e++) {
            const auto [state, action, reward, done, next_state] =
                replay_buffer->sample(batch_size, actor->parameters().back().device());

            torch::Tensor target_q_values;
            {
                torch::NoGradGuard no_grad;

                const auto [next_mu, next_sigma, next_discrete_proba] =
                    actor->act(next_state.vision, next_state.proprioception);

                const auto next_continuous_action = truncated_normal_sample(next_mu, next_sigma);
                const auto next_continuous_entropy = truncated_normal_entropy(next_mu, next_sigma);

                const auto next_discrete_entropy = multinomial_entropy(next_discrete_proba);

                const auto next_target_q_values_1 = target_critic_1->value_per_discrete_action(
                    next_state.vision, next_state.proprioception, next_continuous_action);
                const auto next_target_q_values_2 = target_critic_2->value_per_discrete_action(
                    next_state.vision, next_state.proprioception, next_continuous_action);

                const auto next_min_q_value = torch::sum(
                    next_discrete_proba
                        * torch::min(next_target_q_values_1, next_target_q_values_2),
                    -1, true);

                const auto target_v_value =
                    next_min_q_value
                    + torch::sum(alpha_continuous->alpha() * next_continuous_entropy, -1, true)
                    + torch::sum(alpha_discrete->alpha() * next_discrete_entropy, -1, true);

                target_q_values = reward + (1.f - done.to(torch::kFloat)) * gamma * target_v_value;
            }

            // critic 1
            const auto q_value_1 = critic_1->value_ohe(
                state.vision, state.proprioception, action.continuous_action,
                action.discrete_action);
            const auto critic_1_loss =
                torch::mse_loss(q_value_1, target_q_values, at::Reduction::Mean);

            critic_1_optim->zero_grad();
            critic_1_loss.backward();
            torch::nn::utils::clip_grad_norm_(critic_1->parameters(), GRAD_NORM_MAX);
            critic_1_optim->step();

            // critic 2
            const auto q_value_2 = critic_2->value_ohe(
                state.vision, state.proprioception, action.continuous_action,
                action.discrete_action);
            const auto critic_2_loss =
                torch::mse_loss(q_value_2, target_q_values, at::Reduction::Mean);

            critic_2_optim->zero_grad();
            critic_2_loss.backward();
            torch::nn::utils::clip_grad_norm_(critic_2->parameters(), GRAD_NORM_MAX);
            critic_2_optim->step();

            // target value soft update
            soft_update(target_critic_1, critic_1, tau);
            soft_update(target_critic_2, critic_2, tau);

            // policy
            const auto [curr_mu, curr_sigma, curr_discrete_proba] =
                actor->act(state.vision, state.proprioception);

            const auto curr_continuous_action = truncated_normal_sample(curr_mu, curr_sigma);
            const auto curr_continuous_entropy = truncated_normal_entropy(curr_mu, curr_sigma);

            const auto curr_discrete_entropy = multinomial_entropy(curr_discrete_proba);

            const auto curr_q_values_1 = critic_1->value_per_discrete_action(
                state.vision, state.proprioception, curr_continuous_action);
            const auto curr_q_values_2 = critic_2->value_per_discrete_action(
                state.vision, state.proprioception, curr_continuous_action);
            const auto q_value = torch::sum(
                curr_discrete_proba * torch::min(curr_q_values_1, curr_q_values_2), -1, true);

            const auto actor_loss = -torch::mean(
                torch::sum(alpha_continuous->alpha().detach() * curr_continuous_entropy, -1, true)
                + torch::sum(alpha_discrete->alpha().detach() * curr_discrete_entropy, -1, true)
                + q_value);

            actor_optim->zero_grad();
            actor_loss.backward();
            torch::nn::utils::clip_grad_norm_(actor->parameters(), GRAD_NORM_MAX);
            actor_optim->step();

            // continuous entropy
            const auto curr_continuous_target_entropy =
                truncated_normal_entropy(curr_mu, target_sigma);

            const auto alpha_continuous_loss =
                torch::sum(
                    alpha_continuous->log_alpha()
                        * torch::detach(curr_continuous_entropy - curr_continuous_target_entropy),
                    -1)
                    .mean();

            alpha_continuous_optim->zero_grad();
            alpha_continuous_loss.backward();
            alpha_continuous_optim->step();

            // discrete entropy
            const auto alpha_discrete_loss =
                torch::sum(
                    alpha_discrete->log_alpha()
                        * torch::detach(curr_discrete_entropy - target_discrete_entropy),
                    -1)
                    .mean();

            alpha_discrete_optim->zero_grad();
            alpha_discrete_loss.backward();
            alpha_discrete_optim->step();

            // metrics
            actor_mean_loss_metric->add(actor_loss.cpu().item<float>());
            actor_std_loss_metric->add(actor_loss.cpu().item<float>());

            continuous_entropy_metric->add(curr_continuous_entropy.mean().item<float>());
            discrete_entropy_metric->add(curr_discrete_entropy.mean().item<float>());

            continuous_target_entropy_metric->add(
                curr_continuous_target_entropy.mean().item<float>());
            discrete_target_entropy_metric->add(target_discrete_entropy.item<float>());

            critic_1_mean_loss_metric->add(critic_1_loss.cpu().item<float>());
            critic_1_std_loss_metric->add(critic_1_loss.cpu().item<float>());
            critic_2_mean_loss_metric->add(critic_2_loss.cpu().item<float>());
            critic_2_std_loss_metric->add(critic_2_loss.cpu().item<float>());

            alpha_continuous_metric->add(alpha_continuous->alpha().mean().item<float>());
            alpha_discrete_metric->add(alpha_discrete->alpha().mean().item<float>());
        }

        set_train(false);
    }

    std::vector<std::shared_ptr<AbstractMetric>> SacTrainer::get_metrics() {
        return {
            actor_mean_loss_metric,           actor_std_loss_metric,     critic_1_mean_loss_metric,
            critic_1_std_loss_metric,         critic_2_mean_loss_metric, critic_2_std_loss_metric,
            continuous_target_entropy_metric, continuous_entropy_metric, alpha_continuous_metric,
            discrete_target_entropy_metric,   discrete_entropy_metric,   alpha_discrete_metric};
    }

    void SacTrainer::save(const std::filesystem::path &output_folder) {
        // Models
        save_torch(output_folder, actor, "actor.pt");

        save_torch(output_folder, critic_1, "critic_1.pt");
        save_torch(output_folder, critic_2, "critic_2.pt");

        save_torch(output_folder, target_critic_1, "target_critic_1.pt");
        save_torch(output_folder, target_critic_2, "target_critic_2.pt");

        save_torch(output_folder, alpha_continuous, "alpha_continuous.pt");
        save_torch(output_folder, alpha_discrete, "alpha_discrete.pt");

        // Optimizers
        save_torch(output_folder, actor_optim, "actor_optim.pt");

        save_torch(output_folder, critic_1_optim, "critic_1_optim.pt");
        save_torch(output_folder, critic_2_optim, "critic_2_optim.pt");

        save_torch(output_folder, alpha_continuous_optim, "alpha_continuous_optim.pt");
        save_torch(output_folder, alpha_discrete_optim, "alpha_discrete_optim.pt");

        // string repr
        std::ostringstream actor_repr_oss;
        dump_module_tree(actor, actor_repr_oss, 0, "actor");
        std::ofstream actor_repr_file(output_folder / "actor_repr.txt");
        actor_repr_file << actor_repr_oss.str();
        actor_repr_file.close();

        std::ostringstream critic_repr_oss;
        dump_module_tree(critic_1, critic_repr_oss, 0, "critic");
        std::ofstream critic_repr_file(output_folder / "critic_repr.txt");
        critic_repr_file << critic_repr_oss.str();
        critic_repr_file.close();
    }

    void SacTrainer::set_train(const bool train) const {
        actor->train(train);

        critic_1->train(train);
        critic_2->train(train);

        alpha_continuous->train(train);
        alpha_discrete->train(train);

        // force eval for target critics
        target_critic_1->train(false);
        target_critic_2->train(false);
    }

    void SacTrainer::to(const torch::Device device) {
        actor->to(device);

        critic_1->to(device);
        critic_2->to(device);

        target_critic_1->to(device);
        target_critic_2->to(device);

        alpha_continuous->to(device);
        alpha_discrete->to(device);

        target_sigma = target_sigma.to(device);
        target_discrete_entropy = target_discrete_entropy.to(device);
    }

    int SacTrainer::count_parameters() {
        return count_parameters_impl(actor->parameters())
               + count_parameters_impl(critic_1->parameters())
               + count_parameters_impl(critic_2->parameters())
               + count_parameters_impl(alpha_continuous->parameters())
               + count_parameters_impl(alpha_discrete->parameters());
    }

}// namespace arenai::agent
