//
// Created by samuel on 21/01/2026.
//

#include "./sac.h"

#include "../utils/saver.h"
#include "../utils/target_update.h"
#include "../utils/truncated_normal.h"

SacAgent::SacAgent(
    int nb_sensors, int nb_action, const float learning_rate, int hidden_size_sensors,
    int hidden_size_actions, int actor_hidden_size, int critic_hidden_size,
    const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums, const torch::Device device, int metric_window_size,
    const float tau, const float gamma, const float initial_alpha)
    : actor(std::make_shared<Actor>(
        nb_sensors, nb_action, hidden_size_sensors, actor_hidden_size, vision_channels,
        group_norm_nums)),
      critic_1(std::make_shared<QFunction>(
          nb_sensors, nb_action, hidden_size_sensors, hidden_size_actions, critic_hidden_size,
          vision_channels, group_norm_nums)),
      critic_2(std::make_shared<QFunction>(
          nb_sensors, nb_action, hidden_size_sensors, hidden_size_actions, critic_hidden_size,
          vision_channels, group_norm_nums)),
      target_critic_1(std::make_shared<QFunction>(
          nb_sensors, nb_action, hidden_size_sensors, hidden_size_actions, critic_hidden_size,
          vision_channels, group_norm_nums)),
      target_critic_2(std::make_shared<QFunction>(
          nb_sensors, nb_action, hidden_size_sensors, hidden_size_actions, critic_hidden_size,
          vision_channels, group_norm_nums)),
      alpha_entropy(std::make_shared<AlphaParameter>(initial_alpha)),
      actor_optim(std::make_unique<torch::optim::Adam>(
          actor->parameters(), torch::optim::AdamOptions(learning_rate))),
      critic_1_optim(std::make_unique<torch::optim::Adam>(
          critic_1->parameters(), torch::optim::AdamOptions(learning_rate))),
      critic_2_optim(std::make_unique<torch::optim::Adam>(
          critic_2->parameters(), torch::optim::AdamOptions(learning_rate))),
      entropy_optim(std::make_unique<torch::optim::Adam>(
          alpha_entropy->parameters(), torch::optim::AdamOptions(learning_rate))),
      actor_loss_metric(std::make_shared<Metric>("actor", metric_window_size)),
      critic_1_loss_metric(std::make_shared<Metric>("critic_1", metric_window_size)),
      critic_2_loss_metric(std::make_shared<Metric>("critic_2", metric_window_size)),
      entropy_loss_metric(std::make_shared<Metric>("entropy", metric_window_size, 3, true)),
      entropy_alpha_metric(std::make_shared<Metric>("alpha", metric_window_size)), tau(tau),
      gamma(gamma), target_entropy(truncated_normal_target_entropy(nb_action, -1.f, 1.f)) {

    hard_update(target_critic_1, critic_1);
    hard_update(target_critic_2, critic_2);

    actor->to(device);
    critic_1->to(device);
    critic_2->to(device);
    target_critic_1->to(device);
    target_critic_2->to(device);
    alpha_entropy->to(device);
}

agent_response SacAgent::act(const torch::Tensor &vision, const torch::Tensor &sensors) {
    const auto &[mu, sigma] = actor->act(vision, sensors);
    const auto action = truncated_normal_sample(mu, sigma, -1.f, 1.f);
    return {action, truncated_normal_log_pdf(action, mu, sigma, -1.f, 1.f).sum(1, true)};
}

void SacAgent::train(
    const std::unique_ptr<ReplayBuffer> &replay_buffer, const int epochs, const int batch_size) {

    set_train(true);

    for (int e = 0; e < epochs; e++) {
        const auto [state, action, _, reward, done, next_state] =
            replay_buffer->sample(batch_size, actor->parameters().back().device());

        torch::Tensor target_q_values;
        {
            torch::NoGradGuard no_grad;

            const auto [next_mu, next_sigma] =
                actor->act(next_state.vision, next_state.proprioception);
            const auto next_action = truncated_normal_sample(next_mu, next_sigma, -1.f, 1.f);
            const auto next_log_proba =
                truncated_normal_log_pdf(next_action, next_mu, next_sigma, -1.f, 1.f).sum(-1, true);

            const auto next_target_q_value_1 =
                target_critic_1->value(next_state.vision, next_state.proprioception, next_action);
            const auto next_target_q_value_2 =
                target_critic_2->value(next_state.vision, next_state.proprioception, next_action);

            //const auto normalized_reward = (reward - reward.mean()) / (reward.std() + EPSILON);
            target_q_values = reward
                              + (1.f - done.to(torch::kFloat)) * gamma
                                    * (torch::min(next_target_q_value_1, next_target_q_value_2)
                                       - alpha_entropy->alpha() * next_log_proba);
        }

        // critic 1
        const auto q_value_1 = critic_1->value(state.vision, state.proprioception, action);
        const auto critic_1_loss = torch::mse_loss(q_value_1, target_q_values, at::Reduction::Mean);

        critic_1_optim->zero_grad();
        critic_1_loss.backward();
        critic_1_optim->step();

        // critic 2
        const auto q_value_2 = critic_2->value(state.vision, state.proprioception, action);
        const auto critic_2_loss = torch::mse_loss(q_value_2, target_q_values, at::Reduction::Mean);

        critic_2_optim->zero_grad();
        critic_2_loss.backward();
        critic_2_optim->step();

        // policy
        const auto [curr_mu, curr_sigma] = actor->act(state.vision, state.proprioception);
        const auto curr_action = truncated_normal_sample(curr_mu, curr_sigma, -1.f, 1.f);
        const auto curr_log_proba =
            truncated_normal_log_pdf(curr_action, curr_mu, curr_sigma, -1.f, 1.f).sum(-1, true);

        const auto curr_q_value_1 =
            critic_1->value(state.vision, state.proprioception, curr_action);
        const auto curr_q_value_2 =
            critic_2->value(state.vision, state.proprioception, curr_action);
        const auto q_value = torch::min(curr_q_value_1, curr_q_value_2);

        const auto actor_loss =
            torch::mean(alpha_entropy->alpha().detach() * curr_log_proba - q_value);

        actor_optim->zero_grad();
        actor_loss.backward();
        actor_optim->step();

        // entropy
        const auto entropy_loss =
            -torch::mean(alpha_entropy->log_alpha() * (curr_log_proba.detach() + target_entropy));

        entropy_optim->zero_grad();
        entropy_loss.backward();
        entropy_optim->step();

        // target value soft update
        soft_update(target_critic_1, critic_1, tau);
        soft_update(target_critic_2, critic_2, tau);

        // metrics
        critic_1_loss_metric->add(critic_1_loss.cpu().item<float>());
        critic_2_loss_metric->add(critic_2_loss.cpu().item<float>());
        actor_loss_metric->add(actor_loss.cpu().item<float>());
        entropy_loss_metric->add(entropy_loss.cpu().item<float>());
        entropy_alpha_metric->add(alpha_entropy->alpha().cpu().item<float>());
    }
}

std::vector<std::shared_ptr<Metric>> SacAgent::get_metrics() {
    return {
        actor_loss_metric, critic_1_loss_metric, critic_2_loss_metric, entropy_loss_metric,
        entropy_alpha_metric};
}

void SacAgent::save(const std::filesystem::path &output_folder) {
    // Models
    save_torch(output_folder, actor, "actor.pt");

    save_torch(output_folder, critic_1, "critic_1.pt");
    save_torch(output_folder, critic_2, "critic_2.pt");

    save_torch(output_folder, target_critic_1, "target_critic_1.pt");
    save_torch(output_folder, target_critic_2, "target_critic_2.pt");

    save_torch(output_folder, alpha_entropy, "alpha_entropy.pt");

    export_state_dict_neutral(actor, output_folder / "actor_state_dict");

    // Optimizers
    save_torch(output_folder, actor_optim, "actor_optim.pt");

    save_torch(output_folder, critic_1_optim, "critic_1_optim.pt");
    save_torch(output_folder, critic_2_optim, "critic_2_optim.pt");

    save_torch(output_folder, entropy_optim, "entropy_optim.pt");
}

void SacAgent::set_train(const bool train) {
    actor->train(train);
    critic_1->train(train);
    critic_2->train(train);
    target_critic_1->train(train);
    target_critic_2->train(train);
    alpha_entropy->train(train);
}

void SacAgent::to(const torch::Device device) {
    actor->to(device);
    critic_1->to(device);
    critic_2->to(device);
    target_critic_1->to(device);
    target_critic_2->to(device);
    alpha_entropy->to(device);
}

int SacAgent::count_parameters() {
    return count_parameters_impl(actor->parameters())
           + count_parameters_impl(critic_1->parameters())
           + count_parameters_impl(critic_2->parameters())
           + count_parameters_impl(alpha_entropy->parameters());
}
