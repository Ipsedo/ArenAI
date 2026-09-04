//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_AGENT_HOST_TRAIN_ENVIRONMENT_H
#define ARENAI_AGENT_HOST_TRAIN_ENVIRONMENT_H

#include <arenai_core/environment.h>

#include "../metrics/metric.h"

namespace arenai::agent {

    class TrainTankEnvironment final : public core::BaseTanksEnvironment {
    public:
        TrainTankEnvironment(
            const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend, int nb_tanks,
            const std::filesystem::path &android_assets_path, float wanted_frequency,
            int max_episode_steps, int vision_height, int vision_width, int vision_num_threads);

        std::vector<std::tuple<core::State, core::Reward, core::IsDone>>
        step(float time_delta, const std::vector<core::Action> &actions) override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() const;

        bool is_episode_terminated();

        static void reset_singleton();

    protected:
        void
        on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        void on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

        void
        on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

    private:
        float wanted_frequency;
        int nb_tanks;

        int nb_steps;

        std::vector<bool> done;
        std::vector<bool> already_done;

        int max_episode_steps;

        std::vector<int> nb_hits_per_tanks;
        std::vector<int> nb_kills_per_tanks;

        std::shared_ptr<AbstractMetric> reward_metric;

        // the reward terms the policy trades off, split out of reward_metric
        std::shared_ptr<AbstractMetric> reward_aim_metric;
        std::shared_ptr<AbstractMetric> reward_hit_metric;
        std::shared_ptr<AbstractMetric> reward_received_metric;

        // miss distance of the shells that landed, one sample per step that resolved a shell
        std::shared_ptr<AbstractMetric> miss_distance_metric;

        std::shared_ptr<AbstractMetric> episode_step_mean_nb_metric;

        std::shared_ptr<AbstractMetric> fire_metric;
        std::shared_ptr<AbstractMetric> hit_metric;
        std::shared_ptr<AbstractMetric> kill_metric;

        int nb_kills_episode;

        bool are_all_done();
    };

}// namespace arenai::agent

#endif// ARENAI_AGENT_HOST_TRAIN_ENVIRONMENT_H
