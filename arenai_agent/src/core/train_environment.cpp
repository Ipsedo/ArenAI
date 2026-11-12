#include "./train_environment.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>

#include <arenai_agent/file_reader.h>
#include <arenai_utils/cache.h>
#include <arenai_utils/singleton.h>
#include <arenai_view/backend.h>
#include <arenai_view/camera.h>

#include "../metrics/mean_metric.h"

using namespace arenai;
using namespace arenai::agent;

namespace {
    // the aim potential is in [0, 1]: going from no aim to a perfect one is worth this much
    constexpr float AIM_POTENTIAL_SCALE = 0.3f;
    constexpr float HIT_REWARD_SCALE = 3.f;
    constexpr float RECEIVED_REWARD_SCALE = 1.f;
}// namespace

namespace arenai::agent {

    TrainTankEnvironment::TrainTankEnvironment(
        const std::shared_ptr<view::AbstractGraphicBackend> &graphics_backend, const int nb_tanks,
        const std::filesystem::path &android_assets_path, const float wanted_frequency,
        const int max_episode_steps, const int vision_height, const int vision_width,
        const int vision_num_threads, const int action_repeat, const float gamma)
        : BaseTanksEnvironment(
            std::make_shared<DesktopAssetFileReader>(android_assets_path), graphics_backend,
            nb_tanks, wanted_frequency, vision_height, vision_width, vision_num_threads, false),
          wanted_frequency(wanted_frequency), action_repeat(action_repeat), nb_tanks(nb_tanks),
          nb_steps(0), done(nb_tanks, false), already_done(nb_tanks, false),
          max_episode_steps(max_episode_steps), nb_hits_per_tanks(nb_tanks, 0),
          nb_kills_per_tanks(nb_tanks, 0), reward_metric(std::make_shared<MeanMetric>(
                                               "r", 4 * nb_tanks * max_episode_steps, 1, true)),
          reward_hit_metric(
              std::make_shared<MeanMetric>("r_hit", 4 * nb_tanks * max_episode_steps, 1, true)),
          reward_received_metric(
              std::make_shared<MeanMetric>("r_rcv", 4 * nb_tanks * max_episode_steps, 1, true)),
          reward_aim_metric(
              std::make_shared<MeanMetric>("r_aim", 4 * nb_tanks * max_episode_steps, 1, true)),
          miss_distance_metric(std::make_shared<MeanMetric>("miss", 1024 * nb_tanks, 1)),
          episode_step_mean_nb_metric(std::make_shared<MeanMetric>("s", 32, 1)),
          fire_metric(std::make_shared<MeanMetric>("fire", 256, 2)),
          hit_metric(std::make_shared<MeanMetric>("hit", 16, 2, true)),
          kill_metric(std::make_shared<MeanMetric>("kill", 16, 1)), nb_kills_episode(0),
          nb_fires_episode(0), nb_hits_episode(0),
          aim_shaping(gamma, AIM_POTENTIAL_SCALE, nb_tanks),
          vision_aspect_ratio(
              static_cast<float>(vision_width) / static_cast<float>(vision_height)) {}

    std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>>
    TrainTankEnvironment::step(const float time_delta, const std::vector<core::Action> &actions) {

        already_done = done;

        std::vector rewards(nb_tanks, 0.f);
        std::vector truncated(nb_tanks, false);
        std::vector<model::RewardDetail> reward_details(nb_tanks);
        std::vector shaping_details(nb_tanks, 0.f);
        std::vector nb_fires_per_tank(nb_tanks, 0);
        std::vector nb_hits_per_tank(nb_tanks, 0);

        int nb_frames = 0;

        for (int frame = 0; frame < action_repeat && !is_episode_terminated(); frame++) {
            advance_frame(time_delta, frame == 0 ? actions : core::release_fire(actions));

            // the agents' visions are rendered once, after the window
            const auto frame_results = tick_frame();

            nb_frames++;
            nb_steps++;

            // the tanks' events are consumed every frame: tick() reads them
            apply_on_enemies<int>([&](const auto &tanks) {
                for (int i = 0; i < nb_tanks; i++) {
                    const bool has_hit = tanks[i]->consume_has_hit();
                    const bool has_kill = tanks[i]->consume_has_kill();
                    const bool has_fired = tanks[i]->consume_has_fire();

                    // a tank's rewards stop at its death frame
                    if (done[i]) continue;

                    const auto &[reward, is_done, is_truncated] = frame_results[i];

                    rewards[i] += reward;
                    truncated[i] = is_truncated;

                    const auto detail = tanks[i]->get_last_reward_detail();
                    reward_details[i].hit += detail.hit;
                    reward_details[i].kill += detail.kill;
                    reward_details[i].received += detail.received;
                    reward_details[i].death += detail.death;
                    reward_details[i].nb_fired_shells += detail.nb_fired_shells;
                    reward_details[i].nb_landed_shells += detail.nb_landed_shells;
                    reward_details[i].nb_hits += detail.nb_hits;
                    reward_details[i].sum_miss_distance += detail.sum_miss_distance;

                    if (has_hit) {
                        nb_hits_per_tanks[i]++;
                        nb_hits_per_tank[i]++;
                    }
                    if (has_kill) nb_kills_per_tanks[i]++;
                    if (has_fired) nb_fires_per_tank[i]++;

                    // detect death (kill or suicide)
                    if (is_done) {
                        if (!tanks[i]->is_suicide()) nb_kills_episode++;
                        done[i] = true;
                    }
                }
                return 0;
            });
        }

        apply_on_enemies<int>([&](const auto &tanks) {
            for (int i = 0; i < nb_tanks; i++) {
                if (done[i] || !tanks[i]->is_out_of_shells() || tanks[i]->has_shells_in_flight())
                    continue;
                truncated[i] = true;
                respawn_tank(i);
            }
            return 0;
        });

        // the aim potential is read once per decision, on the state the agent will see
        const auto shaping_rewards = aim_shaping.step(aim_potentials(), done);
        for (int i = 0; i < nb_tanks; i++) {
            if (already_done[i]) continue;
            rewards[i] += shaping_rewards[i] + (HIT_REWARD_SCALE - 1.f) * reward_details[i].hit
                          + (RECEIVED_REWARD_SCALE - 1.f) * reward_details[i].received;
            shaping_details[i] = shaping_rewards[i];
        }

        render_frame(true);
        auto states = read_states();

        // fire frequency (per second, per tank that acted this step)
        int nb_acting = 0, nb_fires = 0, nb_hits = 0;
        for (int i = 0; i < nb_tanks; i++) {
            if (already_done[i]) continue;
            nb_acting++;
            nb_fires += nb_fires_per_tank[i];
            nb_hits += nb_hits_per_tank[i];
        }

        if (nb_acting > 0 && nb_frames > 0)
            fire_metric->add(
                static_cast<float>(nb_fires)
                / (static_cast<float>(nb_acting * nb_frames) * wanted_frequency));

        nb_fires_episode += nb_fires;
        nb_hits_episode += nb_hits;
        step_reward_details = reward_details;
        step_aim_rewards = shaping_details;

        std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>> result;
        result.reserve(nb_tanks);

        // log metrics
        for (int i = 0; i < nb_tanks; i++) {
            if (!already_done[i]) {
                reward_metric->add(rewards[i]);

                const auto &detail = reward_details[i];

                reward_hit_metric->add(HIT_REWARD_SCALE * detail.hit);
                reward_received_metric->add(RECEIVED_REWARD_SCALE * detail.received);
                reward_aim_metric->add(shaping_details[i]);

                if (detail.nb_landed_shells > 0)
                    miss_distance_metric->add(
                        detail.sum_miss_distance / static_cast<float>(detail.nb_landed_shells));

                if (done[i])
                    episode_step_mean_nb_metric->add(
                        static_cast<float>(nb_steps) * wanted_frequency);
            }

            result.emplace_back(std::move(states[i]), rewards[i], done[i], truncated[i]);
        }

        return result;
    }

    void TrainTankEnvironment::on_draw(
        const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) {}

    void TrainTankEnvironment::on_reset_physics(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {

        // close the previous episode's counters (skip the very first reset)
        if (nb_steps > 0) {
            kill_metric->add(static_cast<float>(nb_kills_episode));

            // hit accuracy is undefined on an episode without a single fire
            if (nb_fires_episode > 0)
                hit_metric->add(
                    static_cast<float>(nb_hits_episode) / static_cast<float>(nb_fires_episode));
        }
        nb_kills_episode = 0;

        nb_steps = 0;

        already_done = std::vector(nb_tanks, false);
        done = std::vector(nb_tanks, false);

        nb_hits_per_tanks = std::vector(nb_tanks, 0);
        nb_kills_per_tanks = std::vector(nb_tanks, 0);

        nb_fires_episode = 0;
        nb_hits_episode = 0;

        aim_shaping.reset(aim_potentials());
    }

    std::vector<float> TrainTankEnvironment::aim_potentials() {
        return apply_on_enemies<std::vector<float>>(
            [this](const std::vector<std::shared_ptr<model::EnemyTank>> &tanks) {
                std::vector<float> potentials;
                potentials.reserve(tanks.size());
                for (const auto &tank: tanks)
                    potentials.push_back(tank->get_aim_potential(tanks, vision_aspect_ratio));
                return potentials;
            });
    }

    std::vector<model::RewardDetail> TrainTankEnvironment::last_step_reward_details() const {
        return step_reward_details;
    }

    torch::Tensor TrainTankEnvironment::last_step_aim_rewards() const {
        return torch::tensor(step_aim_rewards, torch::TensorOptions().dtype(torch::kFloat))
            .unsqueeze(-1);
    }

    std::vector<AimGeometry> TrainTankEnvironment::aim_geometry(const float view_half_yaw) {
        return apply_on_enemies<std::vector<AimGeometry>>(
            [view_half_yaw](const std::vector<std::shared_ptr<model::EnemyTank>> &tanks) {
                const float view_half_pitch = view_half_yaw < std::numbers::pi_v<float>
                                                  ? view::DEFAULT_FOV / 2.f
                                                  : std::numbers::pi_v<float>;

                constexpr glm::vec4 origin(0.f, 0.f, 0.f, 1.f);

                std::vector<glm::vec3> positions;
                positions.reserve(tanks.size());
                for (const auto &tank: tanks)
                    positions.emplace_back(tank->get_chassis()->get_model_matrix() * origin);

                std::vector<AimGeometry> result;
                result.reserve(tanks.size());

                for (int i = 0; i < tanks.size(); i++) {
                    AimGeometry geometry{
                        .alive = !tanks[i]->is_dead(),
                        .distance = -1.f,
                        .yaw = 0.f,
                        .pitch = 0.f,
                        .best_angle = std::numbers::pi_v<float>,
                        .best_distance = -1.f,
                        .turret_yaw = 0.f,
                        .view_distance = -1.f,
                        .view_yaw = 0.f,
                        .view_pitch = 0.f,
                        .canon_elevation = 0.f,
                        .canon_relative_elevation = 0.f,
                        .chassis_pitch = 0.f,
                        .chassis_roll = 0.f};

                    const auto canon_matrix = tanks[i]->get_canon()->get_model_matrix();
                    const auto chassis_matrix = tanks[i]->get_chassis()->get_model_matrix();
                    const auto canon_forward_in_chassis =
                        glm::inverse(chassis_matrix) * canon_matrix * glm::vec4(0.f, 0.f, 1.f, 0.f);
                    geometry.turret_yaw =
                        std::atan2(canon_forward_in_chassis.x, canon_forward_in_chassis.z);

                    const auto canon_forward =
                        glm::normalize(glm::vec3(canon_matrix * glm::vec4(0.f, 0.f, 1.f, 0.f)));
                    const auto chassis_forward =
                        glm::normalize(glm::vec3(chassis_matrix * glm::vec4(0.f, 0.f, 1.f, 0.f)));
                    const auto chassis_right =
                        glm::normalize(glm::vec3(chassis_matrix * glm::vec4(1.f, 0.f, 0.f, 0.f)));
                    geometry.canon_elevation = std::asin(std::clamp(canon_forward.y, -1.f, 1.f));
                    geometry.canon_relative_elevation = std::asin(std::clamp(
                        glm::normalize(glm::vec3(canon_forward_in_chassis)).y, -1.f, 1.f));
                    geometry.chassis_pitch = std::asin(std::clamp(chassis_forward.y, -1.f, 1.f));
                    geometry.chassis_roll = std::asin(std::clamp(chassis_right.y, -1.f, 1.f));
                    const auto canon_pos = glm::vec3(canon_matrix * origin);
                    const auto canon_inverse = glm::inverse(canon_matrix);

                    int nearest = -1;
                    for (int j = 0; j < tanks.size(); j++) {
                        if (j == i || tanks[j]->is_dead()) continue;
                        const float distance = glm::length(positions[j] - canon_pos);
                        if (nearest == -1 || distance < geometry.distance) {
                            nearest = j;
                            geometry.distance = distance;
                        }

                        const auto relative =
                            glm::normalize(glm::vec3(canon_inverse * glm::vec4(positions[j], 1.f)));
                        if (const float angle = std::acos(std::clamp(relative.z, -1.f, 1.f));
                            angle < geometry.best_angle) {
                            geometry.best_angle = angle;
                            geometry.best_distance = distance;
                        }

                        const float yaw = std::atan2(relative.x, relative.z);
                        const float pitch = std::asin(std::clamp(relative.y, -1.f, 1.f));
                        if (std::abs(yaw) <= view_half_yaw && std::abs(pitch) <= view_half_pitch
                            && (geometry.view_distance < 0.f
                                || distance < geometry.view_distance)) {
                            geometry.view_distance = distance;
                            geometry.view_yaw = yaw;
                            geometry.view_pitch = pitch;
                        }
                    }

                    if (nearest != -1) {
                        const auto relative =
                            glm::vec3(canon_inverse * glm::vec4(positions[nearest], 1.f));
                        geometry.yaw = std::atan2(relative.x, relative.z);
                        geometry.pitch = std::atan2(
                            relative.y,
                            std::sqrt(relative.x * relative.x + relative.z * relative.z));
                    }

                    result.push_back(geometry);
                }

                return result;
            });
    }

    int TrainTankEnvironment::episode_nb_fires() const { return nb_fires_episode; }

    int TrainTankEnvironment::episode_nb_hits() const { return nb_hits_episode; }

    bool TrainTankEnvironment::are_all_done() {
        return std::accumulate(
            done.begin(), done.end(), true,
            [](const int acc, const bool curr_done) { return acc && curr_done; });
    }

    bool TrainTankEnvironment::is_episode_terminated() {
        return are_all_done() || nb_steps > max_episode_steps;
    }

    void TrainTankEnvironment::on_reset_drawables(
        const std::unique_ptr<model::AbstractPhysicEngine> &engine) {}

    void TrainTankEnvironment::reset_singleton() {
        utils::Singleton<utils::Cache<std::shared_ptr<model::Shape>>>::get_singleton()->clear();
        utils::Singleton<utils::Cache<std::shared_ptr<model::Shape>>>::reset_singleton();
    }

    std::vector<std::shared_ptr<AbstractMetric>> TrainTankEnvironment::get_metrics() const {
        return {
            reward_metric,
            reward_hit_metric,
            reward_received_metric,
            miss_distance_metric,
            episode_step_mean_nb_metric,
            fire_metric,
            hit_metric,
            kill_metric};
    }

    std::vector<std::shared_ptr<AbstractMetric>>
    TrainTankEnvironment::get_csv_only_metrics() const {
        return {reward_aim_metric};
    }

}// namespace arenai::agent
