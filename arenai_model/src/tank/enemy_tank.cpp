//
// Created by samuel on 20/10/2025.
//

#include "./enemy_tank.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <arenai_model/constants.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "../bullet_engine.h"

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
            }),
          tank_prefix_name(tank_prefix_name),
          max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
          curr_frame_upside_down(0), distance_scale(250.f), impact_distance_scale(10.f),
          angle_scale(glm::pi<float>() / 3.f), optimal_distance(75.f), fire_cost(0.3f),
          miss_cost(0.3f), is_dead_already_triggered(false), has_touch(false),
          last_shoot_info(std::nullopt), action_stats(std::make_shared<ActionStats>()) {}

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

    float BulletEnemyTank::compute_hit_reward(
        const glm::vec3 &fire_pos, const glm::vec3 &best_enemy_pos,
        const glm::vec3 &hit_pos) const {
        const float distance_impact = glm::length(hit_pos - best_enemy_pos);

        const glm::vec3 fire_to_enemy = best_enemy_pos - fire_pos;
        const glm::vec3 fire_to_hit = hit_pos - fire_pos;

        const float angle = std::atan2(
            glm::length(glm::cross(fire_to_enemy, fire_to_hit)),
            glm::dot(fire_to_enemy, fire_to_hit));

        const float distance_reward =
            std::exp(-0.5f * std::pow(distance_impact / impact_distance_scale, 2.f));
        const float angle_reward = std::exp(-0.5f * std::pow(angle / angle_scale, 2.f));

        return distance_reward * angle_reward;
    }

    float
    BulletEnemyTank::compute_shoot_reward(const std::vector<std::shared_ptr<EnemyTank>> &tanks) {
        constexpr glm::vec4 world_center(glm::vec3(0.f), 1.f);
        const glm::vec3 chassis_pos = get_chassis()->get_model_matrix() * world_center;

        float best_score = 0.f;
        for (const auto &tank: tanks) {
            if (tank.get() == this) continue;
            if (tank->is_dead() && !tank->is_first_frame_dead()) continue;

            const glm::vec3 other_pos = tank->get_chassis()->get_model_matrix() * world_center;
            const float distance = glm::length(other_pos - chassis_pos);

            const float angle = compute_aim_angle(tank);

            const float distance_score = std::exp(-0.5f * std::pow(distance / distance_scale, 2.f));
            const float angle_score = std::exp(-0.5f * std::pow(angle / angle_scale, 2.f));

            if (const float curr_score = distance_score * angle_score; curr_score > best_score)
                best_score = curr_score;
        }

        return best_score;
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

        // 3. fire cost (anti-spam)
        const auto shoot_reward =
            !is_dead() && action_stats->has_fire() ? compute_shoot_reward(tanks) - fire_cost : 0.f;

        // 4. hit reward
        float hit_reward = 0.f;
        if (last_shoot_info.has_value()) {
            const auto [fire_pos, hit_pos, has_hit, has_killed] = last_shoot_info.value();

            if (const auto best_tank_index = get_nearest_enemy_index(tanks, hit_pos);
                best_tank_index != -1) {
                const auto best_tank_model_matrix =
                    tanks[best_tank_index]->get_chassis()->get_model_matrix();
                const auto best_tank_pos =
                    glm::vec3(best_tank_model_matrix * glm::vec4(glm::vec3(0.f), 1.f));

                const float impact_reward = compute_hit_reward(fire_pos, best_tank_pos, hit_pos);

                hit_reward = impact_reward + (has_hit ? (has_killed ? 2.f : 1.f) : -miss_cost);
            }

            last_shoot_info = std::nullopt;
        }

        // 5. total reward
        const float reward = dead_penalty + shoot_reward + hit_reward;

        return reward;
    }

    float BulletEnemyTank::get_phi(const std::vector<std::shared_ptr<EnemyTank>> &tanks) {
        constexpr glm::vec4 world_center(glm::vec3(0.f), 1.f);
        const glm::vec3 chassis_pos = get_chassis()->get_model_matrix() * world_center;

        std::vector<float> scores;
        std::vector<float> logits;

        for (const auto &enemy: tanks) {
            if (enemy.get() == this || enemy->is_dead()) continue;

            const glm::vec3 enemy_pos = enemy->get_chassis()->get_model_matrix() * world_center;

            const float distance = glm::length(enemy_pos - chassis_pos);
            const float angle = compute_aim_angle(enemy);

            const float distance_score =
                std::exp(-0.5f * std::pow((distance - optimal_distance) / distance_scale, 2.f));
            const float angle_score = (std::cos(angle) + 1.f) / 2.f;

            scores.push_back(distance_score * angle_score);
            logits.push_back(-distance / distance_scale);
        }

        if (scores.empty()) return 0.f;

        const float max_logit = *std::ranges::max_element(logits);
        float sum_exp = 0.f;
        for (const float l: logits) sum_exp += std::exp(l - max_logit);

        float reward = 0.f;
        for (std::size_t i = 0; i < scores.size(); ++i) {
            const float weight = std::exp(logits[i] - max_logit) / sum_exp;
            reward += weight * scores[i];
        }

        return reward;
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

        last_shoot_info = {shell_info.fire_position, shell_info.current_position, hit, killed};
    }

    bool BulletEnemyTank::has_hit_other_tank() {
        if (has_touch) {
            has_touch = false;
            return true;
        }
        return false;
    }

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
