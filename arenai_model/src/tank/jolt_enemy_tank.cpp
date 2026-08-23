//
// Created by samuel on 20/10/2025.
//

#include "./jolt_enemy_tank.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <arenai_model/constants.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "../jolt_engine.h"
#include "./parts/shell.h"

using namespace arenai;
using namespace arenai::model;

namespace {

    // closest point of segment [a, b] to p, and its distance
    float distance_to_segment(
        const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &p, glm::vec3 &closest) {
        const glm::vec3 ab = b - a;
        const float length_squared = glm::length2(ab);

        const float t =
            length_squared > 0.f ? std::clamp(glm::dot(p - a, ab) / length_squared, 0.f, 1.f) : 0.f;

        closest = a + t * ab;

        return glm::length(p - closest);
    }

}// namespace

namespace arenai::model {

    JoltEnemyTank::JoltEnemyTank(
        JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::string &tank_prefix_name, const glm::vec3 chassis_pos,
        const float wanted_frame_frequency)
        : JoltTank(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency,
            [this](const ShellItem *shell, const ShellContactInfo &info, Item *item) {
                on_shell_contact(shell, info, item);
            },
            [this](const std::shared_ptr<ShellItem> &shell) { on_shell_fired(shell); },
            [this] { return nb_shells > 0; }),
          max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
          curr_frame_upside_down(0), miss_distance_scale(1.5f), hit_reward_scale(0.1f),
          hit_received_cost(0.15f), initial_nb_shells(10), nb_shells(initial_nb_shells),
          shells_recharged_per_hit(5),
          nb_frames_per_shell_regen(static_cast<int>(1.5f / wanted_frame_frequency)),
          curr_frame_shell_regen(0), is_dead_already_triggered(false), has_touch(false),
          has_kill(false), has_fired(false) {}

    float JoltEnemyTank::compute_hit_reward(
        const glm::vec3 &fire_pos, const glm::vec3 &enemy_pos, const glm::vec3 &shell_pos) const {

        const glm::vec3 ideal_trajectory = enemy_pos - fire_pos;
        const glm::vec3 miss_trajectory = shell_pos - enemy_pos;

        const float ideal_trajectory_distance = glm::length(ideal_trajectory);
        const float miss_distance = glm::length(miss_trajectory);

        const float ratio =
            miss_distance_scale * miss_distance / std::sqrt(ideal_trajectory_distance);

        return std::exp(-0.5f * std::pow(ratio, 2.f));
    }

    void JoltEnemyTank::update_closest_approach(
        TrackedShell &tracked, const glm::vec3 &shell_pos,
        const std::vector<std::shared_ptr<EnemyTank>> &tanks) const {
        if (const int nearest_index = get_nearest_enemy_index(tanks, shell_pos);
            nearest_index != -1) {

            const auto enemy_pos = glm::vec3(
                tanks[nearest_index]->get_chassis()->get_model_matrix()
                * glm::vec4(glm::vec3(0.f), 1.f));

            // a shell covers ~8 m per frame, an order of magnitude more than the
            // dispersion sigma: sampling positions alone aliases the miss distance,
            // so measure against the segment actually travelled during the frame
            glm::vec3 closest;
            if (const float distance =
                    distance_to_segment(tracked.last_shell_pos, shell_pos, enemy_pos, closest);
                distance < tracked.min_distance) {
                tracked.min_distance = distance;
                tracked.enemy_pos_at_t = enemy_pos;
                tracked.shell_pos_at_t = closest;
                tracked.has_sample = true;
            }
        }

        tracked.last_shell_pos = shell_pos;
    }

    int JoltEnemyTank::get_nearest_enemy_index(
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

    float JoltEnemyTank::get_reward(const std::vector<std::shared_ptr<EnemyTank>> &tanks) {

        // 1. flipped detection
        const auto chassis_model_mat = get_chassis()->get_model_matrix();
        constexpr glm::vec4 up(0.f, 1.f, 0.f, 0.f);
        const auto up_in_chassis = glm::normalize(glm::vec3(chassis_model_mat * up));

        if (const float dot = glm::dot(up_in_chassis, glm::vec3(up)); dot < 0)
            curr_frame_upside_down++;
        else curr_frame_upside_down = 0;

        // 2. passive shell regeneration, capped at the initial reserve: the hit recharge
        // may push the reserve above it, regeneration never does
        if (++curr_frame_shell_regen >= nb_frames_per_shell_regen) {
            curr_frame_shell_regen = 0;
            if (nb_shells < initial_nb_shells) nb_shells++;
        }

        // 3. dead / suicide penalty
        const auto dead_penalty = is_dead() ? -1.f : 0.f;

        // 4. fired shells: sample the closest tank along the trajectory and pay the
        // dispersion gaussian (plus hit/kill bonuses) once the shell dies; firing itself
        // is free — the limited shell reserve (recharged on hit) taxes the spam
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
                // the gaussian stays an order of magnitude under the hit bonus: a gradient
                // toward the aim, not a farmable income; the shell reserve taxes the spam
                shells_reward +=
                    hit_reward_scale
                    * compute_hit_reward(
                        tracked.fire_pos, tracked.enemy_pos_at_t, tracked.shell_pos_at_t);
                if (tracked.has_hit) shells_reward += tracked.has_killed ? 2.f : 0.2f;
            }

            tracked_shells.erase(tracked_shells.begin() + i);
        }

        // 5. hits received penalty
        const float hit_received_penalty =
            -hit_received_cost * static_cast<float>(get_received_hits());

        // 6. total reward
        const float reward = dead_penalty + shells_reward + hit_received_penalty;

        return reward;
    }

    void JoltEnemyTank::on_shell_fired(const std::shared_ptr<ShellItem> &shell) {
        nb_shells--;
        has_fired = true;

        tracked_shells.push_back(
            {.shell = shell,
             .fire_pos = shell->get_fire_position(),
             // seeding the segment at the muzzle also covers the fire → first frame gap
             .last_shell_pos = shell->get_fire_position(),
             .min_distance = std::numeric_limits<float>::infinity(),
             .enemy_pos_at_t = glm::vec3(0.f),
             .shell_pos_at_t = glm::vec3(0.f),
             .has_sample = false,
             .final_shell_pos = glm::vec3(0.f),
             .has_final_pos = false,
             .has_hit = false,
             .has_killed = false});
    }

    void JoltEnemyTank::on_shell_contact(
        const ShellItem *shell, const ShellContactInfo &shell_info, Item *item) {
        for (const auto &i: get_items())
            if (i->get_name() == item->get_name()) return;

        bool hit = false;
        bool killed = false;

        if (const auto &life_item = dynamic_cast<LifeItem *>(item); life_item) {
            if (life_item->is_dead() && !life_item->is_already_dead()) {
                hit = true;
                killed = true;

                has_touch = true;
                has_kill = true;
            } else if (!life_item->is_dead()) {
                hit = true;
                has_touch = true;
            }
        }

        // a shell dies on its first contact, so a hit recharges exactly once
        if (hit) nb_shells += shells_recharged_per_hit;

        for (auto &tracked: tracked_shells) {
            if (tracked.has_final_pos || tracked.shell.lock().get() != shell) continue;

            tracked.final_shell_pos = shell_info.current_position;
            tracked.has_final_pos = true;
            tracked.has_hit = hit;
            tracked.has_killed = killed;
            break;
        }
    }

    bool JoltEnemyTank::consume_has_fire() {
        if (has_fired) {
            has_fired = false;
            return true;
        }
        return false;
    }

    bool JoltEnemyTank::consume_has_hit() {
        if (has_touch) {
            has_touch = false;
            return true;
        }
        return false;
    }

    bool JoltEnemyTank::consume_has_kill() {
        if (has_kill) {
            has_kill = false;
            return true;
        }
        return false;
    }

    // ReSharper disable once CppReferenceToOverriddenVirtualFunction
    bool JoltEnemyTank::is_dead() { return JoltTank::is_dead() || is_suicide(); }

    bool JoltEnemyTank::is_first_frame_dead() { return is_dead() && !is_dead_already_triggered; }

    bool JoltEnemyTank::is_suicide() const {
        return curr_frame_upside_down > max_frames_upside_down;
    }

    void JoltEnemyTank::on_death() {
        if (is_dead() && !is_dead_already_triggered) {
            is_dead_already_triggered = true;
            remove_constraints_from_engine();
            // the wreck stays in the world as an obstacle, but its surviving parts must
            // not pay hits, kills, shells nor survival frames to whoever keeps shooting it
            kill_life_items();
        }
    }

    std::vector<float> JoltEnemyTank::get_proprioception() {
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

        result.push_back(static_cast<float>(nb_shells) / static_cast<float>(initial_nb_shells));

        return result;
    }

}// namespace arenai::model
