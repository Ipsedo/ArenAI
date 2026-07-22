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

        std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>>
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
        int max_frames_without_hit;
        std::vector<int> remaining_frames;
        int nb_frames_added_when_hit;
        int nb_tanks;

        int nb_steps;

        std::vector<bool> done;
        std::vector<bool> already_done;

        int max_episode_steps;

        std::shared_ptr<AbstractMetric> episode_step_mean_nb_metric;
        std::shared_ptr<AbstractMetric> episode_step_std_nb_metric;

        bool only_one_tank_alive();

        bool are_all_done();
    };

}// namespace arenai::agent

#endif// ARENAI_AGENT_HOST_TRAIN_ENVIRONMENT_H
