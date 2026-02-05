//
// Created by samuel on 21/01/2026.
//

#ifndef ARENAI_TRAIN_HOST_SAC_H
#define ARENAI_TRAIN_HOST_SAC_H

#include "../networks/actor.h"
#include "../networks/entropy.h"
#include "../networks/q_function.h"
#include "./agent.h"

class SacAgent : public AbstractAgent {
public:
    SacAgent(
        int nb_sensors, int nb_action, float learning_rate, int hidden_size_sensors,
        int hidden_size_actions, int actor_hidden_size, int critic_hidden_size,
        const std::vector<std::tuple<int, int>> &vision_channels,
        const std::vector<int> &group_norm_nums, torch::Device device, int metric_window_size,
        float tau, float gamma, float initial_alpha);

    void
    train(const std::unique_ptr<ReplayBuffer> &replay_buffer, int epochs, int batch_size) override;
    agent_response act(const torch::Tensor &vision, const torch::Tensor &sensors) override;

    std::vector<std::shared_ptr<Metric>> get_metrics() override;

    void save(const std::filesystem::path &output_folder) override;
    void set_train(bool train) override;
    void to(torch::Device device) override;

    int count_parameters() override;

private:
    std::shared_ptr<Actor> actor;

    std::shared_ptr<QFunction> critic_1;
    std::shared_ptr<QFunction> critic_2;

    std::shared_ptr<QFunction> target_critic_1;
    std::shared_ptr<QFunction> target_critic_2;

    std::shared_ptr<AlphaParameter> alpha_entropy;

    std::shared_ptr<torch::optim::Adam> actor_optim;
    std::shared_ptr<torch::optim::Adam> critic_1_optim;
    std::shared_ptr<torch::optim::Adam> critic_2_optim;
    std::shared_ptr<torch::optim::Adam> entropy_optim;

    std::shared_ptr<Metric> actor_loss_metric;
    std::shared_ptr<Metric> critic_1_loss_metric;
    std::shared_ptr<Metric> critic_2_loss_metric;
    std::shared_ptr<Metric> entropy_loss_metric;
    std::shared_ptr<Metric> entropy_alpha_metric;

    float tau;
    float gamma;
    float target_entropy;
};

#endif//ARENAI_TRAIN_HOST_SAC_H
