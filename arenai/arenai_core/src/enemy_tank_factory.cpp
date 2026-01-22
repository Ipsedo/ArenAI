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
      min_aim_angle(static_cast<float>(M_PI) / 8.f), max_aim_angle(static_cast<float>(M_PI) / 3.f),
      min_distance(5.f), max_distance(100.f), has_touch(false),
      action_stats(std::make_shared<ActionStats>()) {}

float EnemyTankFactory::compute_aim_angle(const std::unique_ptr<EnemyTankFactory> &other_tank) {
    const auto canon_tr = get_canon()->get_model_matrix();
    const auto other_tr = other_tank->get_chassis()->get_model_matrix();

    const auto canon_pos = glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 0.f, 1.f));
    const auto other_pos = glm::vec3(other_tr * glm::vec4(0.f, 0.f, 0.f, 1.f));

    const glm::vec3 to_other = glm::normalize(other_pos - canon_pos);
    const auto forward = glm::normalize(glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 1.f, 0.f)));

    const float d = std::clamp(glm::dot(forward, to_other), -1.f, 1.f);

    return std::acos(d);
}

float EnemyTankFactory::compute_range_reward(const float value, const float min, const float max) {
    return std::clamp((max - value) / (max - min), 0.f, 1.f);
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

    // 3. shaped reward
    const auto chassis_pos = get_chassis()->get_body()->getWorldTransform().getOrigin();

    const float d_min = min_distance;
    const float d_max = max_distance;
    const float band = d_max - d_min;

    const float d_opt = 0.5f * (d_min + d_max);
    const float sigma = 0.5f * band;
    const float inv_sigma2 = 1.0f / (sigma * sigma + EPSILON);

    float shaped_reward = 0.f;
    float shoot_in_aim_reward = 0.f;

    const bool has_shot = action_stats->has_fire();

    for (const auto &other: tank_factories) {
        if (other->tank_prefix_name == tank_prefix_name or other->is_dead()) continue;

        const auto other_pos = other->get_chassis()->get_body()->getWorldTransform().getOrigin();
        const float distance = (chassis_pos - other_pos).length();

        if (!std::isfinite(distance)) continue;

        const float diff = distance - d_opt;
        const float angle = compute_aim_angle(other);

        const float phi_dist = std::exp(-(diff * diff) * inv_sigma2);
        const float phi_angle = 0.5f * (1.f + std::cos(angle));

        const auto curr_shaped_reward = phi_angle * phi_dist;
        shaped_reward = std::max(curr_shaped_reward, shaped_reward);

        const float curr_shoot_in_aim_reward =
            has_shot ? compute_range_reward(angle, min_aim_angle, max_aim_angle) : 0.f;
        shoot_in_aim_reward = std::max(curr_shoot_in_aim_reward, shoot_in_aim_reward);
    }

    const float shoot_penalty = has_shot ? -0.01f : 0.f;
    const float shoot_reward = shoot_penalty + shoot_in_aim_reward;

    // prepare next frame
    const auto reward = hit_reward + dead_penalty + shoot_reward + shaped_reward;
    hit_reward = 0.f;

    // return reward
    return reward;
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

    const auto chassis_pos = chassis->get_body()->getCenterOfMassPosition();

    const auto chassis_vel = chassis->get_body()->getLinearVelocity();

    const auto chassis_tr = chassis->get_body()->getWorldTransform();
    const auto chassis_ang_quat = chassis->get_body()->getOrientation();
    const auto chassis_ang = chassis_ang_quat.getAngle();
    const auto chassis_ang_axis = chassis_ang_quat.getAxis();
    const auto chassis_ang_vel = chassis->get_body()->getAngularVelocity();

    std::vector result{chassis_vel.x(),      chassis_vel.y(),      chassis_vel.z(),
                       chassis_ang,          chassis_ang_axis.x(), chassis_ang_axis.y(),
                       chassis_ang_axis.z(), chassis_ang_vel.x(),  chassis_ang_vel.y(),
                       chassis_ang_vel.z()};
    result.reserve(ENEMY_PROPRIOCEPTION_SIZE);

    for (int i = 1; i < items.size(); i++) {
        const auto body = items[i]->get_body();

        auto pos = body->getCenterOfMassPosition() - chassis_pos;
        auto vel = body->getLinearVelocity() - chassis_vel;

        auto ang_quat = (chassis_tr.inverse() * body->getCenterOfMassTransform()).getRotation();
        auto ang = ang_quat.getAngle();
        auto ang_axis = ang_quat.getAxis();

        auto ang_vel = body->getAngularVelocity() - chassis_ang_vel;

        result.insert(
            result.end(), {pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.z(), ang, ang_axis.x(),
                           ang_axis.y(), ang_axis.z(), ang_vel.x(), ang_vel.y(), ang_vel.z()});
    }
    return result;
}

std::shared_ptr<ActionStats> EnemyTankFactory::get_action_stats() { return action_stats; }
