//
// Created by samuel on 20/10/2025.
//

#include <arenai_core/enemy_tank_factory.h>

EnemyTankFactory::EnemyTankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    const glm::vec3 chassis_pos, const float wanted_frequency)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos), reward(0.f),
      max_frames_upside_down(static_cast<int>(4.f / wanted_frequency)), curr_frame_upside_down(0),
      is_dead_already_triggered(false),
      action_stats(std::make_shared<ActionStats>(3.f, wanted_frequency, 5e-1f)),
      max_frames_without_hit(static_cast<int>(60.f / wanted_frequency)),
      nb_frames_since_last_hit(0) {}

float EnemyTankFactory::get_reward() {
    float actual_reward = reward;

    // prepare next frame
    reward = 0.f;

    const auto chassis = get_items()[0];
    auto chassis_tr = chassis->get_body()->getWorldTransform();
    const btVector3 up(0.f, 1.f, 0.f);
    const btVector3 up_in_chassis = chassis_tr.getBasis() * up;

    if (const btScalar dot = up_in_chassis.normalized().dot(up.normalized()); dot < 0) {
        curr_frame_upside_down++;
        actual_reward -= 0.125f;
    } else curr_frame_upside_down = 0;

    if (is_dead()) actual_reward -= 1.f;

    // actual_reward += 1.125e-1f * action_stats->get_reward();

    /*actual_reward -=
        1.125e-1f * static_cast<float>(std::min(nb_frames_since_last_hit, max_frames_without_hit))
        / static_cast<float>(max_frames_without_hit);*/

    nb_frames_since_last_hit++;

    return actual_reward;
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

std::vector<std::shared_ptr<Item>> EnemyTankFactory::dead_and_get_items() {
    if (is_dead() && !is_dead_already_triggered) {
        is_dead_already_triggered = true;
        return get_items();
    }

    return {};
}

std::vector<float> EnemyTankFactory::get_proprioception() {
    const auto items = get_items();

    const auto &chassis = items[0];

    const auto chassis_pos = chassis->get_body()->getCenterOfMassPosition();
    const auto chassis_pos_norm = std::log(chassis_pos.norm());
    const auto yaw = std::atan2(chassis_pos.x(), chassis_pos.z());
    const auto pitch = std::atan2(
        chassis_pos.y(),
        std::sqrt(std::pow(chassis_pos.x(), 2.f) + std::pow(chassis_pos.z(), 2.f)));

    const auto chassis_vel = chassis->get_body()->getLinearVelocity();

    const auto chassis_ang = chassis->get_body()->getOrientation();
    const auto chassis_ang_vel = chassis->get_body()->getAngularVelocity();

    std::vector result{
        chassis_pos_norm,
        yaw,
        pitch,
        chassis_vel.x(),
        chassis_vel.y(),
        chassis_vel.z(),
        chassis_ang.x(),
        chassis_ang.y(),
        chassis_ang.z(),
        chassis_ang.w(),
        chassis_ang_vel.x(),
        chassis_ang_vel.y(),
        chassis_ang_vel.z()};
    result.reserve((3 * 2 + 4 + 3) * items.size());

    for (int i = 1; i < items.size(); i++) {
        const auto body = items[i]->get_body();

        auto pos = body->getCenterOfMassPosition() - chassis_pos;
        auto vel = body->getLinearVelocity();

        auto ang = body->getCenterOfMassTransform().getRotation();
        auto ang_vel = body->getAngularVelocity();

        result.insert(
            result.end(), {pos.x(), pos.y(), pos.z(), vel.x(), vel.y(), vel.y(), ang.x(), ang.y(),
                           ang.z(), ang.w(), ang_vel.x(), ang_vel.y(), ang_vel.z()});
    }
    return result;
}

std::shared_ptr<ActionStats> EnemyTankFactory::get_action_stats() { return action_stats; }
