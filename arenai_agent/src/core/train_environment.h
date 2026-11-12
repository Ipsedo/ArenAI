#ifndef ARENAI_AGENT_HOST_TRAIN_ENVIRONMENT_H
#define ARENAI_AGENT_HOST_TRAIN_ENVIRONMENT_H

#include <numbers>

#include <torch/torch.h>

#include <arenai_core/environment.h>

#include "../metrics/metric.h"
#include "./potential_shaping.h"

namespace arenai::agent {

    struct AimGeometry {
        bool alive;
        float distance;
        float yaw;
        float pitch;
        float best_angle;
        float best_distance;
        float turret_yaw;
        float view_distance;
        float view_yaw;
        float view_pitch;
        // radians above the horizon (world) and above the chassis plane (relative)
        float canon_elevation;
        float canon_relative_elevation;
        float chassis_pitch;
        float chassis_roll;
    };

    class TrainTankEnvironment final : public core::BaseTanksEnvironment {
    public:
        TrainTankEnvironment(
            const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend, int nb_tanks,
            const std::filesystem::path &android_assets_path, float wanted_frequency,
            int max_episode_steps, int vision_height, int vision_width, int vision_num_threads,
            int action_repeat, float gamma);

        std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>>
        step(float time_delta, const std::vector<core::Action> &actions) override;

        std::vector<std::shared_ptr<AbstractMetric>> get_metrics() const;
        std::vector<std::shared_ptr<AbstractMetric>> get_csv_only_metrics() const;

        bool is_episode_terminated();

        // view_*: the nearest enemy within view_half_yaw of the canon axis (radians)
        std::vector<AimGeometry> aim_geometry(float view_half_yaw = std::numbers::pi_v<float>);

        std::vector<model::RewardDetail> last_step_reward_details() const;

        // [nb_tanks, 1] the aim potential part of the last step's rewards
        torch::Tensor last_step_aim_rewards() const;

        // totals across every tank since the last reset, for the spawn curriculum
        int episode_nb_fires() const;
        int episode_nb_hits() const;

        static void reset_singleton();

    protected:
        void
        on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        void on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

        void
        on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

    private:
        float wanted_frequency;
        int action_repeat;
        int nb_tanks;

        // physics frames since the reset (max_episode_steps is in frames too)
        int nb_steps;

        std::vector<bool> done;
        std::vector<bool> already_done;

        int max_episode_steps;

        std::vector<int> nb_hits_per_tanks;
        std::vector<int> nb_kills_per_tanks;

        std::shared_ptr<AbstractMetric> reward_metric;

        // the reward terms the policy trades off, split out of reward_metric
        std::shared_ptr<AbstractMetric> reward_hit_metric;
        std::shared_ptr<AbstractMetric> reward_received_metric;
        std::shared_ptr<AbstractMetric> reward_aim_metric;

        // miss distance of the shells that landed, one sample per step that resolved a shell
        std::shared_ptr<AbstractMetric> miss_distance_metric;

        std::shared_ptr<AbstractMetric> episode_step_mean_nb_metric;

        std::shared_ptr<AbstractMetric> fire_metric;
        std::shared_ptr<AbstractMetric> hit_metric;
        std::shared_ptr<AbstractMetric> kill_metric;

        int nb_kills_episode;
        int nb_fires_episode;
        int nb_hits_episode;

        std::vector<model::RewardDetail> step_reward_details;
        std::vector<float> step_aim_rewards;

        // the learning term paid on top of the tanks' own event rewards: a dense aim signal
        PotentialShaping aim_shaping;
        float vision_aspect_ratio;

        bool are_all_done();
        std::vector<float> aim_potentials();
    };

}// namespace arenai::agent

#endif// ARENAI_AGENT_HOST_TRAIN_ENVIRONMENT_H
