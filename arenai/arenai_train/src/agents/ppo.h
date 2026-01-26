//
// Created by samuel on 21/01/2026.
//

#ifndef ARENAI_TRAIN_HOST_PPO_H
#define ARENAI_TRAIN_HOST_PPO_H

#include "../networks/actor.h"
#include "../networks/critic.h"
#include "./agent.h"

class PpoAgent : public AbstractAgent {
public:
    PpoAgent(
        int nb_sensors, int nb_action, float learning_rate, int hidden_size_sensors,
        int actor_hidden_size, int critic_hidden_size,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
        float gamma, float epsilon);

    void
    train(const std::unique_ptr<ReplayBuffer> &replay_buffer, int epochs, int batch_size) override;

    agent_response act(const torch::Tensor &vision, const torch::Tensor &sensors) override;

    void set_train(bool train) override;

    std::vector<std::shared_ptr<Metric>> get_metrics() override;

    void save(const std::filesystem::path &output_folder) override;

    void to(torch::Device device) override;

    int count_parameters() override;

private:
    std::shared_ptr<Actor> actor;
    std::shared_ptr<Critic> critic;

    std::shared_ptr<torch::optim::Adam> actor_optim;
    std::shared_ptr<torch::optim::Adam> critic_optim;

    std::shared_ptr<Metric> actor_loss_metric;
    std::shared_ptr<Metric> critic_loss_metric;

    float gamma;
    float epsilon;
};

#endif//ARENAI_TRAIN_HOST_PPO_H
