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
      min_aim_angle(static_cast<float>(M_PI) / 12.f), max_aim_angle(static_cast<float>(M_PI) / 4.f),
      min_distance(5.f), max_distance(100.f),
      optimal_distance(0.5f * (max_distance + min_distance)), sigma_distance(0.25f * max_distance),
      sigma_angle(0.25f * max_aim_angle), softmax_beta(5.f), has_touch(false),
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

float EnemyTankFactory::softmax_scores(const std::vector<float> &scores) const {
    float max_score = -std::numeric_limits<float>::infinity();
    for (const float s: scores) max_score = std::max(s, max_score);

    float numerator = 0.f, denominator = 0.f;

    for (const float q: scores) {
        const float w = std::exp(softmax_beta * q - max_score);

        numerator += q * w;
        denominator += w;
    }
    return denominator > 0.f ? numerator / denominator : 0.f;
}

float EnemyTankFactory::quality_score(const float distance, const float angle) const {
    return std::exp(
        -0.5f * std::pow(angle / sigma_angle, 2.f)
        - 0.5f * std::pow((distance - optimal_distance) / sigma_distance, 2.f));
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
    const bool has_shot = action_stats->has_fire();

    const auto chassis_pos =
        glm::vec3(chassis->get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f));

    std::vector<float> shaped_rewards;

    for (const auto &other: tank_factories) {
        if (other->tank_prefix_name == tank_prefix_name || other->is_dead()) continue;

        const auto other_pos =
            glm::vec3(other->get_chassis()->get_model_matrix() * glm::vec4(glm::vec3(0.f), 1.f));

        const float distance = glm::length(chassis_pos - other_pos);

        if (!std::isfinite(distance)) continue;

        const auto angle = compute_aim_angle(other);

        shaped_rewards.push_back(quality_score(distance, angle));
    }

    const float shaped_reward = softmax_scores(shaped_rewards);
    const float shoot_reward = has_shot ? shaped_reward - 0.1f : 0.f;

    // prepare next frame
    const auto reward =
        0.4f * hit_reward + 0.35f * dead_penalty + 0.15f * shoot_reward + 0.1f * shaped_reward;
    hit_reward = 0.f;

    // return reward
    return reward;
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
