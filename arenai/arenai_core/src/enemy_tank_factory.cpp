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
      curr_frame_upside_down(0), is_dead_already_triggered(false),
      min_aim_angle(static_cast<float>(M_PI) / 12.f), max_aim_angle(static_cast<float>(M_PI) / 3.f),
      min_distance(5.f), max_distance(100.f), has_touch(false),
      action_stats(std::make_shared<ActionStats>()) {}

float EnemyTankFactory::compute_aim_angle(const std::unique_ptr<EnemyTankFactory> &other_tank) {
    const auto canon_tr = get_canon()->get_model_matrix();
    const auto other_tr = other_tank->get_chassis()->get_model_matrix();

    const auto canon_pos = glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 1.7f, 1.f));
    const auto other_pos = glm::vec3(other_tr * glm::vec4(0.f, 0.f, 0.f, 1.f));

    const glm::vec3 to_other = glm::normalize(other_pos - canon_pos);
    const auto forward = glm::normalize(glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 1.f, 0.f)));

    const float d = std::clamp(glm::dot(forward, to_other), -1.f, 1.f);

    return std::acos(d);
}

float EnemyTankFactory::compute_range_reward(const float value, const float min, const float max) {
    return std::clamp((max - value) / (max - min), 0.f, 1.f);
}

float EnemyTankFactory::compute_full_range_reward(
    const float value, const float min, const float max) {

    if (value <= min) return 1.0f;

    if (value <= max) return (max - value) / (max - min);
    if (value <= 2.0f * max) return -(value - max) / max;

    return -1.0f;
}

float EnemyTankFactory::get_reward(
    const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories) {

    // 1. flipped penalty
    const auto chassis = get_chassis();
    auto chassis_tr = chassis->get_body()->getWorldTransform();
    const btVector3 up(0.f, 1.f, 0.f);
    const btVector3 up_in_chassis = chassis_tr.getBasis() * up;

    if (const btScalar dot = up_in_chassis.normalized().dot(up.normalized()); dot < 0)
        curr_frame_upside_down++;
    else curr_frame_upside_down = 0;

    // 2. dead penalty
    const auto dead_penalty = is_dead() ? (is_suicide() ? -0.1f : -1.f) : 0.f;

    // 3. shoot reward
    float max_shoot_reward = 0.f;
    const bool has_shot = action_stats->has_fire();

    const float optimal_distance = 0.5f * (max_distance + min_distance);

    const float band_div = std::pow(max_distance - min_distance, 2.f);
    const float angle_div = std::pow(max_aim_angle - min_aim_angle, 2.f);

    const auto chassis_pos =
        glm::vec3(chassis->get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f));

    for (const auto &other: tank_factories) {
        if (other->tank_prefix_name == tank_prefix_name || other->is_dead()) continue;

        const auto other_pos =
            glm::vec3(other->get_chassis()->get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f));

        const auto clamped_distance =
            std::max(glm::length(chassis_pos - other_pos) - optimal_distance, 0.f);
        const auto angle = compute_aim_angle(other);

        const auto shoot_reward = std::exp(-std::pow(angle, 2.f) / angle_div)
                                  * std::exp(-std::pow(clamped_distance, 2.f) / band_div);

        max_shoot_reward = std::max(shoot_reward, max_shoot_reward);
    }

    constexpr float shoot_penalty = -0.1f;
    const float shoot_reward = has_shot ? max_shoot_reward + shoot_penalty : 0.f;

    // prepare next frame
    const auto reward = 0.6f * hit_reward + 0.3f * dead_penalty + 0.1f * shoot_reward;
    hit_reward = 0.f;

    // return reward
    return reward;
}

float EnemyTankFactory::get_phi(
    const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories) {
    const glm::vec3 chassis_pos =
        get_chassis()->get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f);

    const float band_div = std::pow(max_distance - min_distance, 2.f);
    const float angle_div = std::pow(max_aim_angle - min_aim_angle, 2.f);

    float max_phi = 0.f;

    for (const auto &other: tank_factories) {
        if (other->tank_prefix_name == tank_prefix_name || other->is_dead()) continue;

        const glm::vec3 other_pos =
            other->get_chassis()->get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f);
        const float distance = glm::length(chassis_pos - other_pos);

        if (!std::isfinite(distance)) continue;

        const float angle = compute_aim_angle(other);

        const float phi_dist = std::exp(-std::pow(distance, 2.f) / band_div);
        const float phi_angle = std::exp(-std::pow(angle, 2.f) / angle_div);

        const float phi = 0.2f * phi_dist + 0.3f * phi_angle + 0.5f * phi_angle * phi_dist;

        max_phi = std::max(max_phi, phi);
    }

    return max_phi;
}

void EnemyTankFactory::on_fired_shell_contact(Item *item) {
    for (const auto &i: get_items())
        if (i->get_name() == item->get_name()) return;// self shoot

    if (const auto &life_item = dynamic_cast<LifeItem *>(item); life_item) {
        if (life_item->is_dead() && !life_item->is_already_dead()) {
            hit_reward += 2.0f;
            has_touch = true;
        } else if (!life_item->is_dead()) {
            hit_reward += 1.0f;
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
