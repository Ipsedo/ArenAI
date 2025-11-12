//
// Created by samuel on 20/10/2025.
//

#include <algorithm>
#include <iostream>

#include <arenai_core/enemy_tank_factory.h>

#include <arenai_core/constants.h>

EnemyTankFactory::EnemyTankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    const glm::vec3 chassis_pos, const float wanted_frame_frequency)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency),
      tank_prefix_name(tank_prefix_name), reward(0.f),
      max_frames_upside_down(static_cast<int>(4.f / wanted_frame_frequency)),
      curr_frame_upside_down(0), is_dead_already_triggered(false),
      max_frames_without_hit(static_cast<int>(60.f / wanted_frame_frequency)),
      nb_frames_since_last_hit(0), action_stats(std::make_shared<ActionStats>()),
      min_distance_potential_reward(25.f), max_distance_potential_reward(200.f),
      aim_min_angle_potential_reward(static_cast<float>(M_PI) / 6.f),
      aim_max_angle_potential_reward(static_cast<float>(M_PI) / 2.f) {
}

float EnemyTankFactory::get_reward() {
    float actual_reward = reward;

    // prepare next frame
    reward = 0.f;

    const auto chassis = get_chassis();
    auto chassis_tr = chassis->get_body()->getWorldTransform();
    const btVector3 up(0.f, 1.f, 0.f);
    const btVector3 up_in_chassis = chassis_tr.getBasis() * up;

    if (const btScalar dot = up_in_chassis.normalized().dot(up.normalized()); dot < 0)
        curr_frame_upside_down++;
    else curr_frame_upside_down = 0;

    if (is_dead()) actual_reward -= 1.f;

    // return reward
    return actual_reward;
}

float EnemyTankFactory::get_potential_reward(
    const std::vector<std::unique_ptr<EnemyTankFactory> > &all_enemy_tank_factories) {
    const auto chassis_pos = get_chassis()->get_body()->getWorldTransform().getOrigin();

    // distance
    int nearest_enemy_index = -1;
    float shortest_distance = std::numeric_limits<float>::max();
    float max_distance = 0.f;
    for (int i = 0; i < all_enemy_tank_factories.size(); i++) {
        if (all_enemy_tank_factories[i]->tank_prefix_name != tank_prefix_name) {
            auto other_chassis_pos = all_enemy_tank_factories[i]
                    ->get_chassis()
                    ->get_body()
                    ->getWorldTransform()
                    .getOrigin();
            const float distance = (chassis_pos - other_chassis_pos).length();

            if (distance < shortest_distance) {
                shortest_distance = distance;
                nearest_enemy_index = i;
            }

            max_distance = std::max(max_distance, distance);
        }
    }

    const float reward_distance =
            (max_distance_potential_reward
             - std::clamp(
                 shortest_distance, min_distance_potential_reward, max_distance_potential_reward * 2.f))
            / (max_distance_potential_reward - min_distance_potential_reward);

    // AIM
    const auto canon_tr = get_canon()->get_body()->getWorldTransform();
    const auto other_pos = all_enemy_tank_factories[nearest_enemy_index]
            ->get_chassis()
            ->get_body()
            ->getWorldTransform()
            .getOrigin();

    const btVector3 pos = canon_tr.getOrigin();
    const btVector3 forward = canon_tr.getBasis() * btVector3(0, 0, 1);

    const btVector3 to_target = (other_pos - pos).normalize();

    const btScalar dot = std::clamp(forward.dot(to_target), -1.f, 1.f);
    const btVector3 cross = forward.cross(to_target);
    const btScalar sine = cross.length();

    const auto aim_angle = std::atan2(sine, dot);

    const float aim_reward =
            (aim_max_angle_potential_reward
             - std::clamp(
                 aim_angle, aim_min_angle_potential_reward, aim_max_angle_potential_reward * 2.f))
            / (aim_max_angle_potential_reward - aim_min_angle_potential_reward);

    // fire
    const float fire_reward =
            action_stats->has_fire() ? (aim_angle < aim_min_angle_potential_reward ? 1.f : -1.f) : 0.f;

    // potential reward
    return 1e-1f * fire_reward + 3e-1f * aim_reward + 6e-1f * reward_distance;
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
            nb_frames_since_last_hit = 0;
        } else if (!life_item->is_dead()) {
            reward += 0.5f;
            nb_frames_since_last_hit = 0;
        }
    }
}

bool EnemyTankFactory::is_dead() {
    return TankFactory::is_dead() || curr_frame_upside_down > max_frames_upside_down;
}

std::vector<std::shared_ptr<Item> > EnemyTankFactory::dead_and_get_items() {
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

    std::vector result{
        chassis_vel.x(), chassis_vel.y(), chassis_vel.z(), chassis_ang, chassis_ang_axis.x(),
        chassis_ang_axis.y(), chassis_ang_axis.z(), chassis_ang_vel.x(), chassis_ang_vel.y(),
        chassis_ang_vel.z()
    };
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
            result.end(), {
                pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.y(), ang, ang_axis.x(), ang_axis.y(), ang_axis.z(), ang_vel.x(),
                ang_vel.y(), ang_vel.z()
            });
    }
    return result;
}

std::shared_ptr<ActionStats> EnemyTankFactory::get_action_stats() { return action_stats; }
