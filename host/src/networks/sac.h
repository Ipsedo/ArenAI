//
// Created by samuel on 12/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_SAC_H
#define ARENAI_TRAIN_HOST_SAC_H

#include <filesystem>
#include <memory>

#include "../utils/metric.h"
#include "../utils/replay_buffer.h"
#include "./agent.h"
#include "./entropy.h"

class SacNetworks {
public:
    SacNetworks(
        int nb_sensors, int nb_action, float learning_rate, int hidden_size_sensors,
        int hidden_size_actions, int hidden_size, torch::Device device, int metric_window_size,
        float tau, float gamma);

    void train(const std::unique_ptr<ReplayBuffer> &replay_buffer, int nb_epoch, int batch_size);

    actor_response act(const torch::Tensor &vision, const torch::Tensor &sensors) const;

    std::vector<std::shared_ptr<Metric>> get_metrics() const;

    void save(const std::filesystem::path &output_folder) const;

private:
    std::shared_ptr<SacActor> actor;

    std::shared_ptr<SacCritic> critic_1;
    std::shared_ptr<SacCritic> critic_2;

    std::shared_ptr<SacCritic> target_critic_1;
    std::shared_ptr<SacCritic> target_critic_2;

    std::shared_ptr<AlphaParameter> alpha_entropy;

    torch::optim::Adam actor_optim;
    torch::optim::Adam critic_1_optim;
    torch::optim::Adam critic_2_optim;
    torch::optim::Adam entropy_optim;

    std::shared_ptr<Metric> actor_loss_metric;
    std::shared_ptr<Metric> critic_1_loss_metric;
    std::shared_ptr<Metric> critic_2_loss_metric;
    std::shared_ptr<Metric> entropy_loss_metric;

    float tau;
    float gamma;
    float target_entropy;
};

#endif//ARENAI_TRAIN_HOST_SAC_H
