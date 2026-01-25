//
// Created by samuel on 21/01/2026.
//

#include "./ppo.h"

#include "../utils/saver.h"
#include "../utils/target_update.h"
#include "../utils/truncated_normal.h"
#include "arenai_core/constants.h"

PpoAgent::PpoAgent(
    int nb_sensors, int nb_action, float learning_rate, int hidden_size_sensors,
    int actor_hidden_size, int critic_hidden_size,
    const std::vector<std::tuple<int, int>> &vision_channels,
    const std::vector<int> &group_norm_nums, const torch::Device device, int metric_window_size,
    const float gamma, const float epsilon)
    : old_actor(std::make_shared<Actor>(
        nb_sensors, nb_action, hidden_size_sensors, actor_hidden_size, vision_channels,
        group_norm_nums)),
      actor(std::make_shared<Actor>(
          nb_sensors, nb_action, hidden_size_sensors, actor_hidden_size, vision_channels,
          group_norm_nums)),
      old_critic(std::make_shared<Critic>(
          nb_sensors, hidden_size_sensors, critic_hidden_size, vision_channels, group_norm_nums)),
      critic(std::make_shared<Critic>(
          nb_sensors, hidden_size_sensors, critic_hidden_size, vision_channels, group_norm_nums)),
      actor_optim(std::make_shared<torch::optim::Adam>(actor->parameters(), learning_rate)),
      critic_optim(std::make_shared<torch::optim::Adam>(critic->parameters(), learning_rate)),
      actor_loss_metric(std::make_shared<Metric>("actor", metric_window_size)),
      critic_loss_metric(std::make_shared<Metric>("critic", metric_window_size)), gamma(gamma),
      epsilon(epsilon) {

    old_actor->to(device);
    actor->to(device);

    old_critic->to(device);
    critic->to(device);

    hard_update(old_actor, actor);
    hard_update(old_critic, critic);
}

void PpoAgent::train(
    const std::unique_ptr<ReplayBuffer> &replay_buffer, const int epochs, const int batch_size) {
    set_train(true);

    hard_update(old_actor, actor);
    hard_update(old_critic, critic);

    for (int e = 0; e < epochs; e++) {
        const auto [state, action, reward, done, next_state] =
            replay_buffer->sample(batch_size, actor->parameters().back().device());

        const auto value_old = old_critic->value(state.vision, state.proprioception);
        const auto next_value_old = old_critic->value(next_state.vision, next_state.proprioception);

        const auto target_value =
            torch::detach(reward + (1.f - done.to(torch::kFloat)) * gamma * next_value_old);
        const auto advantage = torch::detach(target_value - value_old);

        // train actor
        const auto &[old_mu, old_sigma] = old_actor->act(state.vision, state.proprioception);
        const auto old_proba = gaussian_tanh_pdf(action, old_mu, old_sigma).sum(1, true).detach();

        const auto &[mu, sigma] = actor->act(state.vision, state.proprioception);
        const auto proba = gaussian_tanh_pdf(action, mu, sigma).sum(1, true);

        const auto r = proba / (old_proba + EPSILON);

        const auto actor_loss =
            torch::mean(advantage * torch::min(r, torch::clamp(r, 1.f - epsilon, 1.f + epsilon)));

        actor_optim->zero_grad();
        actor_loss.backward();
        actor_optim->step();

        // train critic
        const auto value = critic->value(state.vision, state.proprioception);
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
    return {truncated_normal_sample(mu, sigma, -1.f, 1.f)};
}

void PpoAgent::set_train(const bool train) {
    actor->train(train);
    old_critic->train(train);

    critic->train(train);
    old_critic->train(train);
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
    old_actor->to(device);
    actor->to(device);

    old_critic->to(device);
    critic->to(device);
}

int PpoAgent::count_parameters() {
    return count_parameters_impl(actor->parameters()) + count_parameters_impl(critic->parameters());
}
