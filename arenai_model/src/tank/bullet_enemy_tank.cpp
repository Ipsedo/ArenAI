//
// Created by samuel on 20/10/2025.
//

#include "./bullet_enemy_tank.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <arenai_model/constants.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "../bullet_engine.h"
#include "./parts/shell.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    BulletEnemyTank::BulletEnemyTank(
        BulletPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::string &tank_prefix_name, const glm::vec3 chassis_pos,
        const float wanted_frame_frequency)
        : BulletTank(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency,
            [this](const ShellContactInfo &info, Item *item) {
                on_fired_shell_contact(info, item);
            },
            [this](const std::shared_ptr<ShellItem> &shell) { on_shell_fired(shell); }),
          tank_prefix_name(tank_prefix_name),
          max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
          curr_frame_upside_down(0), distance_scale(250.f),
          dispersion_angle_scale(glm::radians(7.5f)), optimal_distance(75.f), fire_cost(0.05f),
          miss_cost(0.1f), is_dead_already_triggered(false), has_touch(false),
          action_stats(std::make_shared<ActionStats>()) {}

    float BulletEnemyTank::compute_aim_angle(const std::shared_ptr<EnemyTank> &other_tank) {
        const auto canon_tr = get_canon()->get_model_matrix();
        const auto other_tr = other_tank->get_chassis()->get_model_matrix();

        const auto canon_muzzle_pos = glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 10.f, 1.f));
        const auto other_pos = glm::vec3(other_tr * glm::vec4(0.f, 0.f, 0.f, 1.f));

        const glm::vec3 to_other = glm::normalize(other_pos - canon_muzzle_pos);
        const auto forward = glm::normalize(glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 1.f, 0.f)));

        const float d = std::clamp(glm::dot(forward, to_other), -1.f, 1.f);

        return std::acos(d);
    }

    float BulletEnemyTank::compute_dispersion_reward(
        const glm::vec3 &fire_pos, const glm::vec3 &enemy_pos, const glm::vec3 &shell_pos) const {
        const glm::vec3 fire_to_enemy = enemy_pos - fire_pos;
        const glm::vec3 fire_to_shell = shell_pos - fire_pos;

        const float angle = std::atan2(
            glm::length(glm::cross(fire_to_enemy, fire_to_shell)),
            glm::dot(fire_to_enemy, fire_to_shell));

        return std::exp(-0.5f * std::pow(angle / dispersion_angle_scale, 2.f));
    }

    void BulletEnemyTank::update_closest_approach(
        TrackedShell &tracked, const glm::vec3 &shell_pos,
        const std::vector<std::shared_ptr<EnemyTank>> &tanks) const {
        const int nearest_index = get_nearest_enemy_index(tanks, shell_pos);
        if (nearest_index == -1) return;

        const auto enemy_pos = glm::vec3(
            tanks[nearest_index]->get_chassis()->get_model_matrix()
            * glm::vec4(glm::vec3(0.f), 1.f));

        if (const float distance = glm::length(shell_pos - enemy_pos);
            distance < tracked.min_distance) {
            tracked.min_distance = distance;
            tracked.enemy_pos_at_t = enemy_pos;
            tracked.shell_pos_at_t = shell_pos;
            tracked.has_sample = true;
        }
    }

    int BulletEnemyTank::get_nearest_enemy_index(
        const std::vector<std::shared_ptr<EnemyTank>> &tanks, const glm::vec3 &pos) const {
        constexpr glm::vec4 world_center(glm::vec3(0.f), 1.f);

        float min_distance = std::numeric_limits<float>::infinity();
        int best_i = -1;

        for (int i = 0; i < tanks.size(); i++) {
            if (tanks[i].get() == this) continue;
            if (tanks[i]->is_dead() && !tanks[i]->is_first_frame_dead()) continue;

            const auto other_pos =
                glm::vec3(tanks[i]->get_chassis()->get_model_matrix() * world_center);

            if (const float distance = glm::length(pos - other_pos); distance < min_distance) {
                min_distance = distance;
                best_i = i;
            }
        }

        return best_i;
    }

    float BulletEnemyTank::get_reward(const std::vector<std::shared_ptr<EnemyTank>> &tanks) {

        // 1. flipped detection
        const auto chassis_model_mat = get_chassis()->get_model_matrix();
        constexpr glm::vec4 up(0.f, 1.f, 0.f, 0.f);
        const auto up_in_chassis = glm::normalize(glm::vec3(chassis_model_mat * up));

        if (const float dot = glm::dot(up_in_chassis, glm::vec3(up)); dot < 0)
            curr_frame_upside_down++;
        else curr_frame_upside_down = 0;

        // 2. dead / suicide penalty
        const auto dead_penalty = is_dead() ? -1.f : 0.f;

        // 3. fired shells: sample the closest tank along the trajectory, pay the
        // dispersion gaussian (plus hit/kill bonuses) once the shell dies
        float shells_reward = 0.f;
        for (int i = static_cast<int>(tracked_shells.size()) - 1; i >= 0; i--) {
            auto &tracked = tracked_shells[i];

            const auto shell = tracked.shell.lock();
            const bool in_flight = shell && !shell->need_destroy();

            if (shell) update_closest_approach(tracked, shell->get_current_position(), tanks);
            else if (tracked.has_final_pos)
                update_closest_approach(tracked, tracked.final_shell_pos, tanks);

            if (in_flight) continue;

            if (tracked.has_sample) {
                shells_reward += compute_dispersion_reward(
                    tracked.fire_pos, tracked.enemy_pos_at_t, tracked.shell_pos_at_t);
                shells_reward += tracked.has_hit ? (tracked.has_killed ? 2.f : 1.f) : -miss_cost;
            }

            tracked_shells.erase(tracked_shells.begin() + i);
        }

        // 4. shoot cost
        const float shoot_cost = action_stats->has_fire() ? -fire_cost : 0.f;

        // 5. total reward
        const float reward = dead_penalty + shells_reward + shoot_cost;

        return reward;
    }

    void BulletEnemyTank::on_shell_fired(const std::shared_ptr<ShellItem> &shell) {
        tracked_shells.push_back(
            {.shell = shell,
             .fire_pos = shell->get_fire_position(),
             .min_distance = std::numeric_limits<float>::infinity(),
             .enemy_pos_at_t = glm::vec3(0.f),
             .shell_pos_at_t = glm::vec3(0.f),
             .has_sample = false,
             .final_shell_pos = glm::vec3(0.f),
             .has_final_pos = false,
             .has_hit = false,
             .has_killed = false});
    }

    void BulletEnemyTank::on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item) {
        for (const auto &i: get_items())
            if (i->get_name() == item->get_name()) return;

        bool hit = false;
        bool killed = false;

        if (const auto &life_item = dynamic_cast<LifeItem *>(item); life_item) {
            if (life_item->is_dead() && !life_item->is_already_dead()) {
                hit = true;
                killed = true;
                has_touch = true;
            } else if (!life_item->is_dead()) {
                hit = true;
                has_touch = true;
            }
        }

        for (auto &tracked: tracked_shells) {
            if (tracked.has_final_pos || tracked.fire_pos != shell_info.fire_position) continue;

            tracked.final_shell_pos = shell_info.current_position;
            tracked.has_final_pos = true;
            tracked.has_hit = hit;
            tracked.has_killed = killed;
            break;
        }
    }

    bool BulletEnemyTank::has_hit_other_tank() {
        if (has_touch) {
            has_touch = false;
            return true;
        }
        return false;
    }

    // ReSharper disable once CppReferenceToOverriddenVirtualFunction
    bool BulletEnemyTank::is_dead() { return BulletTank::is_dead() || is_suicide(); }

    bool BulletEnemyTank::is_first_frame_dead() { return is_dead() && !is_dead_already_triggered; }

    bool BulletEnemyTank::is_suicide() const {
        return curr_frame_upside_down > max_frames_upside_down;
    }

    void BulletEnemyTank::on_death() {
        if (is_dead() && !is_dead_already_triggered) {
            is_dead_already_triggered = true;
            remove_constraints_from_engine();
        }
    }

    std::vector<float> BulletEnemyTank::get_proprioception() {
        const auto items = get_items();

        const auto &chassis = get_chassis();
        const auto chassis_model_matrix = chassis->get_model_matrix();

        const auto chassis_vel = chassis->get_linear_velocity();

        const auto chassis_forward = chassis_model_matrix * glm::vec4(0.f, 0.f, 1.f, 0.f);
        const auto chassis_up = chassis_model_matrix * glm::vec4(0.f, 1.f, 0.f, 0.f);

        const auto chassis_ang_vel = chassis->get_angular_velocity();

        std::vector result{chassis_vel.x,     chassis_vel.y,     chassis_vel.z,
                           chassis_forward.x, chassis_forward.y, chassis_forward.z,
                           chassis_up.x,      chassis_up.y,      chassis_up.z,
                           chassis_ang_vel.x, chassis_ang_vel.y, chassis_ang_vel.z};

        result.reserve(ENEMY_PROPRIOCEPTION_SIZE);

        for (int i = 1; i < items.size(); i++) {
            const auto item_model_matrix = items[i]->get_model_matrix();

            auto relative_model_matrix = glm::inverse(chassis_model_matrix) * item_model_matrix;

            auto pos = relative_model_matrix * glm::vec4(glm::vec3(0.f), 1.f);
            auto vel = items[i]->get_linear_velocity() - chassis_vel;

            auto item_forward = relative_model_matrix * glm::vec4(0.f, 0.f, 1.f, 0.f);
            auto item_up = relative_model_matrix * glm::vec4(0.f, 1.f, 0.f, 0.f);

            auto ang_vel = items[i]->get_angular_velocity() - chassis_ang_vel;

            result.insert(
                result.end(),
                {pos.x, pos.y, pos.z, vel.x, vel.y, vel.z, item_forward.x, item_forward.y,
                 item_forward.z, item_up.x, item_up.y, item_up.z, ang_vel.x, ang_vel.y, ang_vel.z});
        }
        return result;
    }

    std::shared_ptr<ActionStats> BulletEnemyTank::get_action_stats() { return action_stats; }

}// namespace arenai::model
