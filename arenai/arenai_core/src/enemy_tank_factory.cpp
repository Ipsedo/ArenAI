//
// Created by samuel on 20/10/2025.
//

#include <algorithm>
#include <iostream>

#include <arenai_core/constants.h>
#include <arenai_core/enemy_tank_factory.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

EnemyTankFactory::EnemyTankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    const glm::vec3 chassis_pos, const float wanted_frame_frequency)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency),
      tank_prefix_name(tank_prefix_name), hit_reward(0.f),
      max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
      curr_frame_upside_down(0), is_dead_already_triggered(false), distance_scale(300.f),
      optimal_distance(150.f), angle_scale(static_cast<float>(M_PI) / 4.f), has_touch(false),
      action_stats(std::make_shared<ActionStats>()) {}

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

float EnemyTankFactory::angle_quality(const float angle) const {
    return std::exp(-0.5f * std::pow(angle / angle_scale, 2.f));
}

float EnemyTankFactory::distance_quality(const float distance) const {
    return std::exp(-0.5f * std::pow((distance - optimal_distance) / distance_scale, 2.f));
}

float EnemyTankFactory::thresholded_distance_quality(const float distance) const {
    return distance_quality(std::max(distance, optimal_distance));
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

    // 3. shoot penalty / reward
    constexpr auto world_center = glm::vec4(0.f, 0.f, 0.f, 1.f);
    const auto chassis_pos = glm::vec3(chassis_model_mat * world_center);
    float max_quality_score = 0.f;

    for (const auto &other: tank_factories) {
        if (other->tank_prefix_name == tank_prefix_name || other->is_dead()) continue;

        const auto other_pos = glm::vec3(other->get_chassis()->get_model_matrix() * world_center);

        const float distance = glm::length(chassis_pos - other_pos);
        const float angle = compute_aim_angle(other);

        const float quality_score = angle_quality(angle) * thresholded_distance_quality(distance);

        max_quality_score = std::max(quality_score, max_quality_score);
    }

    constexpr float good_fire_reward = 0.2f;
    constexpr float fire_cost = 0.1f;
    const float shoot_reward =
        action_stats->has_fire() ? max_quality_score * good_fire_reward - fire_cost : 0.f;

    // 4. total reward
    const float reward = hit_reward + dead_penalty + shoot_reward;
    hit_reward = 0.f;

    return reward;
}

float EnemyTankFactory::get_phi(
    const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories) {

    constexpr glm::vec4 world_center(glm::vec3(0.f), 1.f);

    const auto chassis_pos = glm::vec3(get_chassis()->get_model_matrix() * world_center);

    float max_quality_score = 0.f;

    for (const auto &other: tank_factories) {
        if (other->tank_prefix_name == tank_prefix_name || other->is_dead()) continue;

        const auto other_pos = glm::vec3(other->get_chassis()->get_model_matrix() * world_center);

        const float distance = glm::length(chassis_pos - other_pos);
        const float angle = compute_aim_angle(other);

        const float angle_score = angle_quality(angle);
        const float distance_score = distance_quality(distance);

        const float quality_score =
            0.5f * angle_score * distance_score + 0.3f * angle_score + 0.2f * distance_score;

        max_quality_score = std::max(quality_score, max_quality_score);
    }

    return max_quality_score;
}

void EnemyTankFactory::on_fired_shell_contact(Item *item) {
    for (const auto &i: get_items())
        if (i->get_name() == item->get_name()) return;// self shoot

    if (const auto &life_item = dynamic_cast<LifeItem *>(item); life_item) {
        if (life_item->is_dead() && !life_item->is_already_dead()) {
            hit_reward += 1.0f;
            has_touch = true;
        } else if (!life_item->is_dead()) {
            hit_reward += 0.5f;
            has_touch = true;
        }
    }
}

bool EnemyTankFactory::has_shoot_other_tank() {
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

std::shared_ptr<ActionStats> EnemyTankFactory::get_action_stats() { return action_stats; }
