#include "./jolt_enemy_tank.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <arenai_model/constants.h>

#include "../geometry.h"
#include "../jolt_engine.h"
#include "./parts/canon.h"
#include "./parts/shell.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    JoltEnemyTank::JoltEnemyTank(
        JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::string &tank_prefix_name, const glm::vec3 chassis_pos,
        const float wanted_frame_frequency, const ShellEconomy &shell_economy)
        : JoltTank(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency,
            ENEMY_TURRET_RADIAL_VELOCITY, ENEMY_CANON_RADIAL_VELOCITY,
            [this](const ShellItem *shell, const ShellContactInfo &info, Item *item) {
                on_shell_contact(shell, info, item);
            },
            [this](const std::shared_ptr<ShellItem> &shell) { on_shell_fired(shell); },
            [this] {
                const bool can_fire = get_can_fire();
                if (can_fire) curr_cooldown_frame = 0;
                return can_fire;
            }),
          max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
          curr_frame_upside_down(0), aim_tolerance_min(2.5f),
          aim_tolerance_angle(glm::radians(3.f)), initial_nb_shells(shell_economy.initial_shells),
          nb_shells(initial_nb_shells), max_shells(shell_economy.max_shells),
          fire_cooldown_frames(static_cast<int>(1.f / 6.f / wanted_frame_frequency)),
          curr_cooldown_frame(fire_cooldown_frames),
          shells_recharged_per_hit(shell_economy.shells_per_hit), is_dead_already_triggered(false),
          has_hit(false), has_kill(false), has_fired(false), killed_nb(0), hit_nb(0),
          nb_fired_to_pay(0) {}

    int JoltEnemyTank::get_score() const { return killed_nb * 10 + hit_nb; }

    void JoltEnemyTank::update_closest_approach(
        TrackedShell &tracked, const glm::vec3 &shell_pos,
        const std::vector<std::shared_ptr<EnemyTank>> &tanks) const {
        if (const int nearest_index = get_nearest_enemy_index(tanks, shell_pos);
            nearest_index != -1) {

            const auto enemy_pos = glm::vec3(
                tanks[nearest_index]->get_chassis()->get_model_matrix()
                * glm::vec4(glm::vec3(0.f), 1.f));

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

    float JoltEnemyTank::compute_aim_quality(
        const glm::vec3 &fire_pos, const glm::vec3 &enemy_pos, const glm::vec3 &shell_pos) const {
        const float shot_distance = glm::length(enemy_pos - fire_pos);
        const float miss_distance = glm::length(shell_pos - enemy_pos);

        const float tolerance = aim_tolerance_min + aim_tolerance_angle * shot_distance;
        const float ratio = miss_distance / tolerance;

        return 1.f / (1.f + ratio * ratio);
    }

    float JoltEnemyTank::predict_aim(
        const glm::vec3 &fire_pos, const glm::vec3 &fire_velocity,
        const std::vector<std::shared_ptr<EnemyTank>> &tanks) const {
        constexpr float time_step = 0.1f;
        constexpr float max_flight_time = 20.f;
        auto excluded = get_body_ids();
        for (const auto &other: tracked_shells)
            if (const auto shell = other.shell.lock())
                excluded.push_back(shell->get_body()->GetID());

        TrackedShell predicted{
            .last_shell_pos = fire_pos,
            .min_distance = std::numeric_limits<float>::infinity(),
            .has_sample = false};

        glm::vec3 pos = fire_pos;
        glm::vec3 velocity = fire_velocity;

        for (float t = 0.f; t < max_flight_time; t += time_step) {
            const glm::vec3 next = pos + velocity * time_step
                                   + glm::vec3(0.f, -0.5f * GRAVITY * time_step * time_step, 0.f);
            velocity.y -= GRAVITY * time_step;

            const auto ray_hit = engine.ray_test_body(pos, next, excluded);
            const glm::vec3 end = ray_hit ? pos + ray_hit->fraction * (next - pos) : next;

            update_closest_approach(predicted, end, tanks);

            if (ray_hit) break;
            pos = end;
        }

        if (!predicted.has_sample) return 0.f;

        return compute_aim_quality(fire_pos, predicted.enemy_pos_at_t, predicted.shell_pos_at_t);
    }

    float
    JoltEnemyTank::get_aim_potential(const std::vector<std::shared_ptr<EnemyTank>> &tanks) const {
        if (is_dead()) return 0.f;

        const auto canon = std::dynamic_pointer_cast<CanonItem>(get_canon());
        const auto [position, direction] = canon->get_fire_pose();

        return predict_aim(position, direction * SHELL_SPEED, tanks);
    }

    float JoltEnemyTank::get_reward() const {
        RewardDetail detail;

        // 1. death penalty: killed or flipped over, the two ways a tank ends before the episode
        if (is_dead()) detail.death = -DEATH_PENALTY;

        // 2. landed shells: the health fraction they removed, and the kill
        for (int i = static_cast<int>(tracked_shells.size()) - 1; i >= 0; i--) {
            auto &tracked = tracked_shells[i];

            const auto shell = tracked.shell.lock();

            if (const bool in_flight = shell && !shell->need_destroy(); in_flight) continue;

            if (tracked.has_sample) {
                detail.nb_landed_shells++;
                detail.sum_miss_distance +=
                    glm::length(tracked.shell_pos_at_t - tracked.enemy_pos_at_t);

                if (tracked.has_hit) {
                    detail.nb_hits++;
                    detail.hit += DAMAGE_REWARD * tracked.damage_fraction;
                    if (tracked.has_killed) detail.kill += KILL_REWARD;
                }
            }
        }

        // 3. shells fired, counted once on their fire frame
        detail.nb_fired_shells = nb_fired_to_pay;
        nb_fired_to_pay = 0;

        // 4. damage received: the health fraction lost on each part
        for (const auto life_item: get_life_items())
            for (const auto &impact: life_item->consume_hits_received())
                detail.received -= DAMAGE_REWARD * impact.damages / life_item->initial_health();

        // 5. total reward, kept split for the metrics
        last_reward_detail = detail;

        return detail.hit + detail.kill + detail.received + detail.death;
    }

    RewardDetail JoltEnemyTank::get_last_reward_detail() const { return last_reward_detail; }

    void JoltEnemyTank::tick(const std::vector<std::shared_ptr<EnemyTank>> &tanks) {
        // 1. flipped detection
        const auto chassis_model_mat = get_chassis()->get_model_matrix();
        constexpr glm::vec4 up(0.f, 1.f, 0.f, 0.f);
        const auto up_in_chassis = glm::normalize(glm::vec3(chassis_model_mat * up));

        if (const float dot = glm::dot(up_in_chassis, glm::vec3(up)); dot < 0)
            curr_frame_upside_down++;
        else curr_frame_upside_down = 0;

        nb_shells = std::min(nb_shells, max_shells);

        // 2. track shells
        for (int i = static_cast<int>(tracked_shells.size()) - 1; i >= 0; i--) {
            auto &tracked = tracked_shells[i];

            // reward already saw it, can remove
            if (tracked.need_remove) {
                tracked_shells.erase(tracked_shells.begin() + i);
                continue;
            }

            const auto shell = tracked.shell.lock();
            const bool in_flight = shell && !shell->need_destroy();

            if (shell) update_closest_approach(tracked, shell->get_current_position(), tanks);
            else if (tracked.has_final_pos)
                update_closest_approach(tracked, tracked.final_shell_pos, tanks);

            // remove next tick => allow get_reward(...) to see it
            if (!in_flight) tracked.need_remove = true;
        }

        // 3. fire cooldown
        curr_cooldown_frame = std::min(fire_cooldown_frames, curr_cooldown_frame + 1);
    }

    void JoltEnemyTank::on_shell_fired(const std::shared_ptr<ShellItem> &shell) {
        nb_shells--;
        has_fired = true;
        nb_fired_to_pay++;

        tracked_shells.push_back(
            {.shell = shell,
             .last_shell_pos = shell->get_fire_position(),
             .min_distance = std::numeric_limits<float>::infinity(),
             .enemy_pos_at_t = glm::vec3(0.f),
             .shell_pos_at_t = glm::vec3(0.f),
             .has_sample = false,
             .final_shell_pos = glm::vec3(0.f),
             .has_final_pos = false,
             .damage_fraction = 0.f,
             .has_hit = false,
             .has_killed = false,
             .need_remove = false});
    }

    void JoltEnemyTank::on_shell_contact(
        const ShellItem *shell, const ShellContactInfo &shell_info, Item *item) {
        for (const auto &i: get_items())
            if (i->get_name() == item->get_name()) return;

        bool hit = false;
        bool killed = false;
        float damage_fraction = 0.f;

        if (const auto &life_item = dynamic_cast<LifeItem *>(item); life_item) {
            if (life_item->is_dead() && !life_item->is_already_dead()) {
                hit = true;
                killed = true;

                has_hit = true;
                has_kill = true;
                killed_nb++;
            } else if (!life_item->is_dead()) {
                hit = true;
                has_hit = true;
                hit_nb++;
            }

            if (hit) damage_fraction = shell_info.damages / life_item->initial_health();
        }

        if (hit) nb_shells += shells_recharged_per_hit;

        for (auto &tracked: tracked_shells) {
            if (tracked.has_final_pos || tracked.shell.lock().get() != shell) continue;

            tracked.final_shell_pos = shell_info.current_position;
            tracked.has_final_pos = true;
            tracked.damage_fraction = damage_fraction;
            tracked.has_hit = hit;
            tracked.has_killed = killed;
            break;
        }
    }

    bool JoltEnemyTank::get_can_fire() const {
        return nb_shells > 0 && curr_cooldown_frame == fire_cooldown_frames;
    }

    bool JoltEnemyTank::consume_has_fire() {
        if (has_fired) {
            has_fired = false;
            return true;
        }
        return false;
    }

    bool JoltEnemyTank::consume_has_hit() {
        if (has_hit) {
            has_hit = false;
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
    bool JoltEnemyTank::is_dead() const { return JoltTank::is_dead() || is_suicide(); }

    bool JoltEnemyTank::is_first_frame_dead() const {
        return is_dead() && !is_dead_already_triggered;
    }

    bool JoltEnemyTank::is_suicide() const {
        return curr_frame_upside_down > max_frames_upside_down;
    }

    void JoltEnemyTank::on_death() {
        if (is_dead() && !is_dead_already_triggered) {
            is_dead_already_triggered = true;

            remove_constraints_from_engine();
            kill_life_items();
        }
    }

    bool JoltEnemyTank::is_out_of_shells() const { return nb_shells <= 0; }

    bool JoltEnemyTank::has_shells_in_flight() const {
        return std::ranges::any_of(tracked_shells, [](const TrackedShell &tracked) {
            const auto shell = tracked.shell.lock();
            return shell && !shell->need_destroy();
        });
    }

    void JoltEnemyTank::respawn(const glm::vec3 chassis_pos) {
        teleport(chassis_pos);
        restore_life_items();

        curr_frame_upside_down = 0;
        nb_shells = initial_nb_shells;
        curr_cooldown_frame = fire_cooldown_frames;
        tracked_shells.clear();
        nb_fired_to_pay = 0;
    }
    std::vector<float> JoltEnemyTank::get_proprioception() const {
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

        for (const auto &life_item: get_life_items()) result.push_back(life_item->health_ratio());

        return result;
    }

    float JoltEnemyTank::get_remaining_shells_ratio() const {
        return static_cast<float>(nb_shells) / static_cast<float>(max_shells);
    }

    float JoltEnemyTank::get_cooldown_ratio() const {
        return static_cast<float>(curr_cooldown_frame) / static_cast<float>(fire_cooldown_frames);
    }

}// namespace arenai::model
