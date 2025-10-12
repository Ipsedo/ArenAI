//
// Created by samuel on 12/10/2025.
//

#include "./sac.h"

#include <phyvr_core/types.h>

#include "../utils/saver.h"
#include "./target_update.h"
#include "./truncated_normal.h"

SacNetworks::SacNetworks(
    int nb_sensors, int nb_action, const float learning_rate, int hidden_size_sensors,
    int hidden_size, const torch::Device device, int metric_window_size, const float tau,
    const float gamma)
    : actor(std::make_shared<SacActor>(nb_sensors, nb_action, hidden_size_sensors, hidden_size)),
      critic_1(
          std::make_shared<SacCritic>(nb_sensors, nb_action, hidden_size_sensors, hidden_size)),
      critic_2(
          std::make_shared<SacCritic>(nb_sensors, nb_action, hidden_size_sensors, hidden_size)),
      target_critic_1(
          std::make_shared<SacCritic>(nb_sensors, nb_action, hidden_size_sensors, hidden_size)),
      target_critic_2(
          std::make_shared<SacCritic>(nb_sensors, nb_action, hidden_size_sensors, hidden_size)),
      alpha_entropy(std::make_shared<AlphaParameter>(1.f)),
      actor_optim(
          torch::optim::Adam(actor->parameters(), torch::optim::AdamOptions(learning_rate))),
      critic_1_optim(
          torch::optim::Adam(critic_1->parameters(), torch::optim::AdamOptions(learning_rate))),
      critic_2_optim(
          torch::optim::Adam(critic_2->parameters(), torch::optim::AdamOptions(learning_rate))),
      entropy_optim(torch::optim::Adam(
          alpha_entropy->parameters(), torch::optim::AdamOptions(learning_rate))),
      actor_loss_metric(std::make_shared<Metric>("actor", metric_window_size)),
      critic_1_loss_metric(std::make_shared<Metric>("critic_1", metric_window_size)),
      critic_2_loss_metric(std::make_shared<Metric>("critic_2", metric_window_size)),
      entropy_loss_metric(std::make_shared<Metric>("entropy", metric_window_size)), tau(tau),
      gamma(gamma), target_entropy(-static_cast<float>(nb_action)) {

    hard_update(target_critic_1, critic_1);
    hard_update(target_critic_2, critic_2);

    actor->to(device);
    critic_1->to(device);
    critic_2->to(device);
    target_critic_1->to(device);
    target_critic_2->to(device);
    alpha_entropy->to(device);
}

actor_response SacNetworks::act(const torch::Tensor &vision, const torch::Tensor &sensors) const {
    return actor->act(vision, sensors);
}

void SacNetworks::train(
    const std::shared_ptr<ReplayBuffer> &replay_buffer, const int nb_epoch, const int batch_size) {
    for (int e = 0; e < nb_epoch; e++) {
        const auto [state, action, reward, done, next_state] =
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

            const auto target_v_value = torch::min(next_target_q_value_1, next_target_q_value_2)
                                        - alpha_entropy->alpha() * next_log_proba;

            target_q_values = reward + (1.f - done.to(torch::kFloat)) * gamma * target_v_value;
        }

        // critic 1
        const auto q_value_1 = critic_1->value(state.vision, state.proprioception, action);
        const auto critic_1_loss = torch::mse_loss(q_value_1, target_q_values, at::Reduction::Mean);

        critic_1_optim.zero_grad();
        critic_1_loss.backward();
        critic_1_optim.step();

        // critic 2
        const auto q_value_2 = critic_2->value(state.vision, state.proprioception, action);
        const auto critic_2_loss = torch::mse_loss(q_value_2, target_q_values, at::Reduction::Mean);

        critic_2_optim.zero_grad();
        critic_2_loss.backward();
        critic_2_optim.step();

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

        actor_optim.zero_grad();
        actor_loss.backward();
        actor_optim.step();

        // entropy
        const auto entropy_loss =
            -torch::mean(alpha_entropy->log_alpha() * (curr_log_proba.detach() + target_entropy));

        entropy_optim.zero_grad();
        entropy_loss.backward();
        entropy_optim.step();

        // target value soft update
        soft_update(target_critic_1, critic_1, tau);
        soft_update(target_critic_2, critic_2, tau);

        // metrics
        actor_loss_metric->add(actor_loss.cpu().item().toFloat());
        critic_1_loss_metric->add(critic_1_loss.cpu().item().toFloat());
        critic_2_loss_metric->add(critic_2_loss.cpu().item().toFloat());
        entropy_loss_metric->add(entropy_loss.cpu().item().toFloat());
    }
}

std::vector<std::shared_ptr<Metric>> SacNetworks::get_metrics() const {
    return {actor_loss_metric, critic_1_loss_metric, critic_2_loss_metric, entropy_loss_metric};
}

void SacNetworks::save(const std::filesystem::path &output_folder) const {
    save_torch(output_folder, actor, "actor.pt");

    save_torch(output_folder, critic_1, "critic_1.pt");
    save_torch(output_folder, critic_2, "critic_2.pt");

    save_torch(output_folder, target_critic_1, "target_critic_1.pt");
    save_torch(output_folder, target_critic_2, "target_critic_2.pt");

    save_torch(output_folder, alpha_entropy, "alpha_entropy.pt");

    export_state_dict_neutral(actor, output_folder / "actor_state_dict");
}
