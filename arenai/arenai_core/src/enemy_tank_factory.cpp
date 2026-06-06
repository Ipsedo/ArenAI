//
// Created by samuel on 20/10/2025.
//

#include <algorithm>
#include <cmath>
#include <vector>

#include <arenai_core/constants.h>
#include <arenai_core/enemy_tank_factory.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <iostream>
#include <ostream>

#include <glm/gtx/norm.hpp>

EnemyTankFactory::EnemyTankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    const glm::vec3 chassis_pos, const float wanted_frame_frequency)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency),
      tank_prefix_name(tank_prefix_name),
      max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
      curr_frame_upside_down(0), distance_scale(150.f), is_dead_already_triggered(false),
      has_touch(false), last_shoot_info(std::nullopt) {}

float EnemyTankFactory::compute_aim_angle(const std::unique_ptr<EnemyTankFactory> &other_tank) {
    const auto canon_tr = get_canon()->get_model_matrix();
    const auto other_tr = other_tank->get_chassis()->get_model_matrix();

    const auto canon_muzzle_pos = glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 10.f, 1.f));
    const auto other_pos = glm::vec3(other_tr * glm::vec4(0.f, 0.f, 0.f, 1.f));

    const glm::vec3 to_other = glm::normalize(other_pos - canon_muzzle_pos);
    const auto forward = glm::normalize(glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 1.f, 0.f)));

    const float d = std::clamp(glm::dot(forward, to_other), -1.f, 1.f);

    return std::acos(d);
}

float EnemyTankFactory::compute_shoot_reward(
    const glm::vec3 &fire_pos, const glm::vec3 &best_enemy_pos, const glm::vec3 &hit_pos) {
    const glm::vec3 enemy_vector = best_enemy_pos - fire_pos;
    const float enemy_len2 = glm::dot(enemy_vector, enemy_vector);

    const float t =
        enemy_len2 > 0.0f ? glm::dot(hit_pos - fire_pos, enemy_vector) / enemy_len2 : 0.0f;
    const glm::vec3 proj_hit_pos = fire_pos + t * enemy_vector;

    const glm::vec3 proj_hit_vector = proj_hit_pos - fire_pos;
    const glm::vec3 proj_hit_pos_to_hit_pos = hit_pos - proj_hit_pos;

    const float hit_vector_length = glm::length(proj_hit_vector);
    const float enemy_vector_length = glm::length(enemy_vector);
    const float min_length = std::max(hit_vector_length, enemy_vector_length);

    const float distance_reward =
        min_length > 0.0f ? glm::dot(enemy_vector, proj_hit_vector) / (min_length * min_length)
                          : 0.0f;

    const float angle_dispersion =
        std::atan2(glm::length(proj_hit_pos_to_hit_pos), hit_vector_length);
    const float angle_reward = 1.0f - 2.0f * angle_dispersion / glm::pi<float>();

    return distance_reward * angle_reward;
}

float EnemyTankFactory::get_reward(
    const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories) {

    // 1. flipped detection
    const auto chassis_model_mat = get_chassis()->get_model_matrix();
    constexpr glm::vec4 up(0.f, 1.f, 0.f, 0.f);
    const auto up_in_chassis = glm::normalize(glm::vec3(chassis_model_mat * up));

    if (const float dot = glm::dot(up_in_chassis, glm::vec3(up)); dot < 0) curr_frame_upside_down++;
    else curr_frame_upside_down = 0;

    // 2. dead / suicide penalty
    const auto dead_penalty = is_dead() ? (is_suicide() ? -0.5f : -1.f) : 0.f;

    // 3. shoot reward
    float shoot_reward = 0.f;
    if (const auto [best_tank_index, _] = get_best_score(tank_factories);
        last_shoot_info.has_value() && best_tank_index != -1) {

        constexpr float shoot_penalty = 0.25f;

        const auto [fire_pos, hit_pos, has_hit, has_killed] = last_shoot_info.value();

        const auto best_tank_model_matrix =
            tank_factories[best_tank_index]->get_chassis()->get_model_matrix();
        const auto best_tank_pos =
            glm::vec3(best_tank_model_matrix * glm::vec4(glm::vec3(0.f), 1.f));

        // shoot reward
        shoot_reward = compute_shoot_reward(fire_pos, best_tank_pos, hit_pos)
                       + (has_hit ? 0.5f : 0.f) + (has_killed ? 1.f : 0.f) - shoot_penalty;

        last_shoot_info = std::nullopt;
    }

    // 4. total reward
    const float reward = dead_penalty + shoot_reward;

    return reward;
}

std::tuple<int, float> EnemyTankFactory::get_best_score(
    const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories) {
    constexpr glm::vec4 world_center(glm::vec3(0.f), 1.f);
    const auto chassis_model_matrix = get_chassis()->get_model_matrix();
    const auto chassis_pos = glm::vec3(chassis_model_matrix * world_center);

    float best_score = -1.f;
    int best_i = -1;

    for (int i = 0; i < tank_factories.size(); i++) {
        if (tank_factories[i]->tank_prefix_name == tank_prefix_name || tank_factories[i]->is_dead())
            continue;

        const auto other_pos =
            glm::vec3(tank_factories[i]->get_chassis()->get_model_matrix() * world_center);

        const float distance = glm::length(chassis_pos - other_pos);
        const float distance_score = std::exp(-0.5f * std::pow(distance / distance_scale, 2.f));

        const float angle = compute_aim_angle(tank_factories[i]);
        const float angle_score = (std::cos(angle) + 1.f) / 2.f;

        if (const float score = distance_score * angle_score; score > best_score) {
            best_score = score;
            best_i = i;
        }
    }

    return {best_i, best_score};
}

void EnemyTankFactory::on_fired_shell_contact(ShellItem *shell, Item *item) {
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

    last_shoot_info = {shell->get_fire_position(), shell->get_current_position(), hit, killed};
}

bool EnemyTankFactory::has_hit_other_tank() {
    if (has_touch) {
        has_touch = false;
        return true;
    }
    return false;
}

bool EnemyTankFactory::is_dead() { return TankFactory::is_dead() || is_suicide(); }

bool EnemyTankFactory::is_suicide() const {
    return curr_frame_upside_down > max_frames_upside_down;
}

std::vector<std::shared_ptr<Item>> EnemyTankFactory::dead_and_get_items() {
    if (is_dead() && !is_dead_already_triggered) {
        is_dead_already_triggered = true;
        return get_items();
    }

    return {};
}

std::vector<float> EnemyTankFactory::get_proprioception() {
    const auto items = get_items();

    const auto &chassis = get_chassis();
    const auto chassis_model_matrix = chassis->get_model_matrix();

    const auto chassis_vel = chassis->get_body()->getLinearVelocity();

    const auto chassis_forward = chassis_model_matrix * glm::vec4(0.f, 0.f, 1.f, 0.f);
    const auto chassis_up = chassis_model_matrix * glm::vec4(0.f, 1.f, 0.f, 0.f);

    const auto chassis_ang_vel = chassis->get_body()->getAngularVelocity();

    std::vector result{chassis_vel.x(),     chassis_vel.y(),     chassis_vel.z(),
                       chassis_forward.x,   chassis_forward.y,   chassis_forward.z,
                       chassis_up.x,        chassis_up.y,        chassis_up.z,
                       chassis_ang_vel.x(), chassis_ang_vel.y(), chassis_ang_vel.z()};

    result.reserve(ENEMY_PROPRIOCEPTION_SIZE);

    for (int i = 1; i < items.size(); i++) {
        const auto body = items[i]->get_body();
        const auto item_model_matrix = items[i]->get_model_matrix();

        auto relative_model_matrix = glm::inverse(chassis_model_matrix) * item_model_matrix;

        auto pos = relative_model_matrix * glm::vec4(glm::vec3(0.f), 1.f);
        auto vel = body->getLinearVelocity() - chassis_vel;

        auto item_forward = relative_model_matrix * glm::vec4(0.f, 0.f, 1.f, 0.f);
        auto item_up = relative_model_matrix * glm::vec4(0.f, 1.f, 0.f, 0.f);

        auto ang_vel = body->getAngularVelocity() - chassis_ang_vel;

        result.insert(
            result.end(), {pos.x, pos.y, pos.z, vel.x(), vel.y(), vel.z(), item_forward.x,
                           item_forward.y, item_forward.z, item_up.x, item_up.y, item_up.z,
                           ang_vel.x(), ang_vel.y(), ang_vel.z()});
    }
    return result;
}
