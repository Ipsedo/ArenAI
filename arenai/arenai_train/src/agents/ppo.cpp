//
// Created by samuel on 21/01/2026.
//

#include "./ppo.h"

#include "../distributions/truncated_normal.h"
#include "../utils/saver.h"
#include "../utils/target_update.h"
#include "arenai_core/constants.h"

PpoAgent::PpoAgent(
    int nb_sensors, int nb_action, float learning_rate, int hidden_size_sensors,
    int actor_hidden_size, int critic_hidden_size,
    const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums, const torch::Device device, int metric_window_size,
    const float gamma, const float epsilon)
    : actor(std::make_shared<Actor>(
        nb_sensors, nb_action, hidden_size_sensors, actor_hidden_size, vision_channels,
        group_norm_nums)),
      critic(std::make_shared<Critic>(
          nb_sensors, hidden_size_sensors, critic_hidden_size, vision_channels, group_norm_nums)),
      actor_optim(std::make_shared<torch::optim::Adam>(actor->parameters(), learning_rate)),
      critic_optim(std::make_shared<torch::optim::Adam>(critic->parameters(), learning_rate)),
      actor_loss_metric(std::make_shared<Metric>("actor", metric_window_size)),
      critic_loss_metric(std::make_shared<Metric>("critic", metric_window_size)), gamma(gamma),
      epsilon(epsilon) {

    actor->to(device);
    critic->to(device);
}

void PpoAgent::train(
    const std::unique_ptr<ReplayBuffer> &replay_buffer, const int epochs, const int batch_size) {
    set_train(true);

    for (int e = 0; e < epochs; e++) {
        const auto [state, action, old_log_proba, reward, done, next_state] =
            replay_buffer->sample(batch_size, actor->parameters().back().device());

        const auto value = critic->value(state.vision, state.proprioception);
        const auto next_value = critic->value(next_state.vision, next_state.proprioception);

        const auto target_value =
            torch::detach(reward + (1.f - done.to(torch::kFloat)) * gamma * next_value);
        const auto advantage = torch::detach(target_value - value);

        // train actor
        const auto &[mu, sigma] = actor->act(state.vision, state.proprioception);
        const auto log_proba = truncated_normal_log_pdf(action, mu, sigma, -1.f, 1.f).sum(1, true);

        const auto ratio = torch::exp(log_proba - old_log_proba);
        const auto clipped_ratio = torch::clamp(ratio, 1.f - epsilon, 1.f + epsilon);

        const auto actor_loss =
            -torch::mean(torch::sum(torch::min(ratio * advantage, clipped_ratio * advantage), 1));

        actor_optim->zero_grad();
        actor_loss.backward();
        actor_optim->step();

        // train critic
        const auto critic_loss = torch::mse_loss(value, target_value);

        critic_optim->zero_grad();
        critic_loss.backward();
        critic_optim->step();

        // metrics
        actor_loss_metric->add(actor_loss.item<float>());
        critic_loss_metric->add(critic_loss.item<float>());
    }
}

agent_response PpoAgent::act(const torch::Tensor &vision, const torch::Tensor &sensors) {
    const auto &[mu, sigma] = actor->act(vision, sensors);
    const auto action = truncated_normal_sample(mu, sigma, -1.f, 1.f);
    return {action, truncated_normal_log_pdf(action, mu, sigma, -1.f, 1.f).sum(1, true)};
}

void PpoAgent::set_train(const bool train) {
    actor->train(train);
    critic->train(train);
}

std::vector<std::shared_ptr<Metric>> PpoAgent::get_metrics() {
    return {actor_loss_metric, critic_loss_metric};
}

void PpoAgent::save(const std::filesystem::path &output_folder) {
    save_torch(output_folder, actor, "actor.pt");
    save_torch(output_folder, critic, "critic.pt");

    save_torch(output_folder, actor_optim, "actor_optim.pt");
    save_torch(output_folder, critic_optim, "critic_optim.pt");

    export_state_dict_neutral(actor, output_folder / "actor_state_dict");
}

void PpoAgent::to(const torch::Device device) {
    actor->to(device);
    critic->to(device);
}

int PpoAgent::count_parameters() {
    return count_parameters_impl(actor->parameters()) + count_parameters_impl(critic->parameters());
}
