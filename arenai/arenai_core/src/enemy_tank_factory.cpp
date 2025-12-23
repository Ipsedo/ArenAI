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
      tank_prefix_name(tank_prefix_name), reward(0.f),
      max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
      curr_frame_upside_down(0), is_dead_already_triggered(false),
      min_distance_potential_reward(10.f), max_distance_potential_reward(100.f),
      aim_min_angle_potential_reward(static_cast<float>(M_PI) / 6.f),
      aim_max_angle_potential_reward(static_cast<float>(M_PI) / 3.f), has_touch(false) {}

float EnemyTankFactory::get_reward() {
    float actual_reward = reward;

    // prepare next frame
    reward = 0.f;

    // flipped penalty
    const auto chassis = get_chassis();
    auto chassis_tr = chassis->get_body()->getWorldTransform();
    const btVector3 up(0.f, 1.f, 0.f);
    const btVector3 up_in_chassis = chassis_tr.getBasis() * up;

    if (const btScalar dot = up_in_chassis.normalized().dot(up.normalized()); dot < 0)
        curr_frame_upside_down++;
    else curr_frame_upside_down = 0;

    // dead penalty
    if (is_dead()) {
        if (is_suicide()) actual_reward = -0.5f;
        else actual_reward = -1.f;
    }

    // return reward
    return actual_reward;
}

float EnemyTankFactory::compute_value_range_reward(
    const float value, const float min_value, const float max_value) {
    const float max_penalty_value = max_value * 2.f;

    // bounds
    if (value <= min_value) return 1.f;
    if (value >= max_penalty_value) return -1.f;

    // [min_value, max_value] : 1 -> 0
    if (value <= max_value) return (max_value - value) / (max_value - min_value);

    // [max_value, 2 * max_value] : 0 -> -1
    return -(value - max_value) / (max_penalty_value - max_value);
}

float EnemyTankFactory::compute_aim_angle(const std::unique_ptr<EnemyTankFactory> &other_tank) {
    const auto canon_tr = get_canon()->get_model_matrix();
    const glm::vec3 other_pos =
        other_tank->get_chassis()->get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f);

    const glm::vec3 pos = canon_tr * glm::vec4(glm::vec3(0.f), 1.f);

    const glm::vec3 forward = glm::normalize(glm::vec3(canon_tr * glm::vec4(0.f, 0.f, 1.f, 0.f)));
    const glm::vec3 to_target = glm::normalize(other_pos - pos);

    const float dot = std::clamp(glm::dot(forward, to_target), -1.f, 1.f);

    const glm::vec3 cross = glm::cross(forward, to_target);
    const float sine = glm::length(cross);

    return std::atan2(sine, dot);
}

float EnemyTankFactory::get_potential_reward(
    const std::vector<std::unique_ptr<EnemyTankFactory>> &all_enemy_tank_factories) {
    const auto chassis_pos = get_chassis()->get_body()->getWorldTransform().getOrigin();

    // distance
    int nearest_enemy_index = -1;
    float shortest_distance = std::numeric_limits<float>::max();
    for (int i = 0; i < all_enemy_tank_factories.size(); i++) {
        if (all_enemy_tank_factories[i]->tank_prefix_name != tank_prefix_name) {
            auto other_chassis_pos = all_enemy_tank_factories[i]
                                         ->get_chassis()
                                         ->get_body()
                                         ->getWorldTransform()
                                         .getOrigin();

            if (const float distance = (chassis_pos - other_chassis_pos).length();
                distance < shortest_distance) {
                shortest_distance = distance;
                nearest_enemy_index = i;
            }
        }
    }

    const float distance_reward = compute_value_range_reward(
        shortest_distance, min_distance_potential_reward, max_distance_potential_reward);

    // AIM
    const auto aim_angle = compute_aim_angle(all_enemy_tank_factories[nearest_enemy_index]);
    const float aim_reward =
        compute_value_range_reward(
            aim_angle, aim_min_angle_potential_reward, aim_max_angle_potential_reward)
        * (distance_reward + 1.f) / 2.f;

    // potential reward
    return 0.7f * aim_reward + 0.3f * distance_reward;
}

void EnemyTankFactory::on_fired_shell_contact(Item *item) {
    bool self_shoot = false;
    for (const auto &i: get_items()) {
        if (i->get_name() == item->get_name()) {
            self_shoot = true;
            break;
        }
    }

    if (const auto &life_item = dynamic_cast<LifeItem *>(item); !self_shoot && life_item) {
        if (life_item->is_dead() && !life_item->is_already_dead()) {
            reward += 1.0f;
            has_touch = true;
        } else if (!life_item->is_dead()) {
            reward += 0.5f;
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

bool EnemyTankFactory::is_dead() {
    return TankFactory::is_dead() || curr_frame_upside_down > max_frames_upside_down;
}

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
            result.end(), {pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.y(), ang, ang_axis.x(),
                           ang_axis.y(), ang_axis.z(), ang_vel.x(), ang_vel.y(), ang_vel.z()});
    }
    return result;
}
