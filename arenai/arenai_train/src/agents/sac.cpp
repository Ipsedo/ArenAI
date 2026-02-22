//
// Created by samuel on 21/01/2026.
//

#include "./sac.h"

#include <fstream>

#include "../distributions/multinomial.h"
#include "../distributions/truncated_normal.h"
#include "../utils/print_module.h"
#include "../utils/saver.h"
#include "../utils/target_update.h"

SacAgent::SacAgent(
    int nb_sensors, int nb_continuous_actions, int nb_discrete_actions, const float learning_rate,
    int hidden_size_sensors, int hidden_size_actions, int actor_hidden_size, int critic_hidden_size,
    const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums, const torch::Device device, int metric_window_size,
    const float tau, const float gamma, const float initial_alpha)
    : actor(std::make_shared<Actor>(
        nb_sensors, nb_continuous_actions, nb_discrete_actions, hidden_size_sensors,
        actor_hidden_size, vision_channels, group_norm_nums)),
      critic_1(std::make_shared<QFunction>(
          nb_sensors, nb_continuous_actions + nb_discrete_actions, hidden_size_sensors,
          hidden_size_actions, critic_hidden_size, vision_channels, group_norm_nums)),
      critic_2(std::make_shared<QFunction>(
          nb_sensors, nb_continuous_actions + nb_discrete_actions, hidden_size_sensors,
          hidden_size_actions, critic_hidden_size, vision_channels, group_norm_nums)),
      target_critic_1(std::make_shared<QFunction>(
          nb_sensors, nb_continuous_actions + nb_discrete_actions, hidden_size_sensors,
          hidden_size_actions, critic_hidden_size, vision_channels, group_norm_nums)),
      target_critic_2(std::make_shared<QFunction>(
          nb_sensors, nb_continuous_actions + nb_discrete_actions, hidden_size_sensors,
          hidden_size_actions, critic_hidden_size, vision_channels, group_norm_nums)),
      alpha_continuous(std::make_shared<AlphaParameter>(initial_alpha)),
      alpha_discrete(std::make_shared<AlphaParameter>(initial_alpha)),
      actor_optim(std::make_unique<torch::optim::Adam>(
          actor->parameters(), torch::optim::AdamOptions(learning_rate))),
      critic_1_optim(std::make_unique<torch::optim::Adam>(
          critic_1->parameters(), torch::optim::AdamOptions(learning_rate))),
      critic_2_optim(std::make_unique<torch::optim::Adam>(
          critic_2->parameters(), torch::optim::AdamOptions(learning_rate))),
      alpha_continuous_optim(std::make_unique<torch::optim::Adam>(
          alpha_continuous->parameters(), torch::optim::AdamOptions(learning_rate))),
      alpha_discrete_optim(std::make_unique<torch::optim::Adam>(
          alpha_discrete->parameters(), torch::optim::AdamOptions(learning_rate))),
      actor_loss_metric(std::make_shared<Metric>("actor", metric_window_size)),
      critic_1_loss_metric(std::make_shared<Metric>("critic_1", metric_window_size)),
      critic_2_loss_metric(std::make_shared<Metric>("critic_2", metric_window_size)),
      continuous_entropy_metric(std::make_shared<Metric>("entropy_c", metric_window_size)),
      discrete_entropy_metric(std::make_shared<Metric>("entropy_d", metric_window_size)),
      alpha_continuous_metric(std::make_shared<Metric>("alpha_c", metric_window_size)),
      alpha_discrete_metric(std::make_shared<Metric>("alpha_d", metric_window_size)), tau(tau),
      gamma(gamma), continous_target_entropy(
                        truncated_normal_target_entropy(nb_continuous_actions, -1.f, 1.f, 1.f)),
      discrete_target_entropy(0.98f * std::log(nb_discrete_actions)) {

    hard_update(target_critic_1, critic_1);
    hard_update(target_critic_2, critic_2);

    actor->to(device);
    critic_1->to(device);
    critic_2->to(device);
    target_critic_1->to(device);
    target_critic_2->to(device);
    alpha_continuous->to(device);
    alpha_discrete->to(device);
}

agent_response SacAgent::act(const torch::Tensor &vision, const torch::Tensor &sensors) {
    const auto &[mu, sigma, discrete] = actor->act(vision, sensors);

    const auto continuous_action = truncated_normal_sample(mu, sigma, -1.f, 1.f);
    const auto continuous_log_proba =
        truncated_normal_log_pdf(continuous_action, mu, sigma, -1.f, 1.f).sum(-1, true);

    const auto discrete_action = gumbel_hard(discrete);
    const auto discrete_log_proba = multinomial_log_proba(discrete_action, discrete);

    return {continuous_action, continuous_log_proba, discrete_action, discrete_log_proba};
}

void SacAgent::train(
    const std::unique_ptr<ReplayBuffer> &replay_buffer, const int epochs, const int batch_size) {

    set_train(true);

    for (int e = 0; e < epochs; e++) {
        const auto [state, action, reward, done, next_state] =
            replay_buffer->sample(batch_size, actor->parameters().back().device());

        torch::Tensor target_q_values;
        {
            torch::NoGradGuard no_grad;

            const auto [next_mu, next_sigma, next_discrete] =
                actor->act(next_state.vision, next_state.proprioception);

            const auto next_continuous_action =
                truncated_normal_sample(next_mu, next_sigma, -1.f, 1.f);
            const auto next_continuous_entropy =
                -truncated_normal_log_pdf(next_continuous_action, next_mu, next_sigma, -1.f, 1.f)
                     .sum(-1, true);

            const auto next_discrete_action = gumbel_hard(next_discrete);
            const auto next_discrete_entropy = multinomial_entropy(next_discrete);

            const auto next_action = torch::cat({next_continuous_action, next_discrete_action}, -1);

            const auto next_target_q_value_1 =
                target_critic_1->value(next_state.vision, next_state.proprioception, next_action);
            const auto next_target_q_value_2 =
                target_critic_2->value(next_state.vision, next_state.proprioception, next_action);

            target_q_values = reward
                              + (1.f - done.to(torch::kFloat)) * gamma
                                    * (torch::min(next_target_q_value_1, next_target_q_value_2)
                                       + alpha_continuous->alpha() * next_continuous_entropy
                                       + alpha_discrete->alpha() * next_discrete_entropy);
        }

        const auto concat_action =
            torch::cat({action.continuous_action, action.discrete_action}, -1);

        // critic 1
        const auto q_value_1 = critic_1->value(state.vision, state.proprioception, concat_action);
        const auto critic_1_loss = torch::mse_loss(q_value_1, target_q_values, at::Reduction::Mean);

        critic_1_optim->zero_grad();
        critic_1_loss.backward();
        critic_1_optim->step();

        // critic 2
        const auto q_value_2 = critic_2->value(state.vision, state.proprioception, concat_action);
        const auto critic_2_loss = torch::mse_loss(q_value_2, target_q_values, at::Reduction::Mean);

        critic_2_optim->zero_grad();
        critic_2_loss.backward();
        critic_2_optim->step();

        // policy
        const auto [curr_mu, curr_sigma, curr_discrete] =
            actor->act(state.vision, state.proprioception);

        const auto curr_continuous_action = truncated_normal_sample(curr_mu, curr_sigma, -1.f, 1.f);
        const auto curr_continuous_entropy =
            -truncated_normal_log_pdf(curr_continuous_action, curr_mu, curr_sigma, -1.f, 1.f)
                 .sum(-1, true);

        const auto curr_discrete_action = gumbel_hard(curr_discrete);
        const auto curr_discrete_entropy = multinomial_entropy(curr_discrete);

        const auto curr_action = torch::cat({curr_continuous_action, curr_discrete_action}, -1);

        const auto curr_q_value_1 =
            critic_1->value(state.vision, state.proprioception, curr_action);
        const auto curr_q_value_2 =
            critic_2->value(state.vision, state.proprioception, curr_action);
        const auto q_value = torch::min(curr_q_value_1, curr_q_value_2);

        const auto actor_loss = -torch::mean(
            alpha_continuous->alpha().detach() * curr_continuous_entropy
            + alpha_discrete->alpha().detach() * curr_discrete_entropy + q_value);

        actor_optim->zero_grad();
        actor_loss.backward();
        actor_optim->step();

        // continuous entropy
        const auto alpha_continuous_loss = torch::mean(
            alpha_continuous->log_alpha()
            * (curr_continuous_entropy.detach() - continous_target_entropy));

        alpha_continuous_optim->zero_grad();
        alpha_continuous_loss.backward();
        alpha_continuous_optim->step();

        // discrete entropy
        const auto alpha_discrete_loss = torch::mean(
            alpha_discrete->log_alpha()
            * (curr_discrete_entropy.detach() - discrete_target_entropy));

        alpha_discrete_optim->zero_grad();
        alpha_discrete_loss.backward();
        alpha_discrete_optim->step();

        // target value soft update
        soft_update(target_critic_1, critic_1, tau);
        soft_update(target_critic_2, critic_2, tau);

        // metrics
        critic_1_loss_metric->add(critic_1_loss.cpu().item<float>());
        critic_2_loss_metric->add(critic_2_loss.cpu().item<float>());

        actor_loss_metric->add(actor_loss.cpu().item<float>());

        continuous_entropy_metric->add(curr_continuous_entropy.mean().item<float>());
        discrete_entropy_metric->add(curr_discrete_entropy.mean().item<float>());

        alpha_continuous_metric->add(alpha_continuous->alpha().item<float>());
        alpha_discrete_metric->add(alpha_discrete->alpha().item<float>());
    }
}

std::vector<std::shared_ptr<Metric>> SacAgent::get_metrics() {
    return {actor_loss_metric,         critic_1_loss_metric,    critic_2_loss_metric,
            continuous_entropy_metric, alpha_continuous_metric, discrete_entropy_metric,
            alpha_discrete_metric};
}

void SacAgent::save(const std::filesystem::path &output_folder) {
    // Models
    save_torch(output_folder, actor, "actor.pt");

    save_torch(output_folder, critic_1, "critic_1.pt");
    save_torch(output_folder, critic_2, "critic_2.pt");

    save_torch(output_folder, target_critic_1, "target_critic_1.pt");
    save_torch(output_folder, target_critic_2, "target_critic_2.pt");

    save_torch(output_folder, alpha_continuous, "alpha_continuous.pt");
    save_torch(output_folder, alpha_discrete, "alpha_discrete.pt");

    export_state_dict_neutral(actor, output_folder / "actor_state_dict");

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

void SacAgent::set_train(const bool train) {
    actor->train(train);
    critic_1->train(train);
    critic_2->train(train);
    target_critic_1->train(train);
    target_critic_2->train(train);
    alpha_continuous->train(train);
    alpha_discrete->train(train);
}

void SacAgent::to(const torch::Device device) {
    actor->to(device);
    critic_1->to(device);
    critic_2->to(device);
    target_critic_1->to(device);
    target_critic_2->to(device);
    alpha_continuous->to(device);
    alpha_discrete->to(device);
}

int SacAgent::count_parameters() {
    return count_parameters_impl(actor->parameters())
           + count_parameters_impl(critic_1->parameters())
           + count_parameters_impl(critic_2->parameters())
           + count_parameters_impl(alpha_continuous->parameters())
           + count_parameters_impl(alpha_discrete->parameters());
}

float SacAgent::get_continuous_target_entropy() const { return continous_target_entropy; }

float SacAgent::get_discrete_target_entropy() const { return discrete_target_entropy; }
