//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H
#define ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H

#include <arenai_model/tank_factory.h>

#include "./action_stats.h"

class EnemyTankFactory final : public TankFactory {
public:
    EnemyTankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        glm::vec3 chassis_pos, float wanted_frame_frequency);

    float get_reward(const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories);
    float
    get_potential_reward(const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories);

    bool is_dead() override;
    bool is_suicide() const;

    bool has_shoot_other_tank();

    std::vector<std::shared_ptr<Item>> dead_and_get_items();

    std::vector<float> get_proprioception();

    std::shared_ptr<ActionStats> get_action_stats();

protected:
    void on_fired_shell_contact(Item *item) override;

private:
    std::string tank_prefix_name;

    float hit_reward;
    int max_frames_upside_down;
    int curr_frame_upside_down;

    bool is_dead_already_triggered;

    float min_aim_angle_reward;
    float max_aim_angle_reward;
    float min_distance_reward;
    float max_distance_reward;

    bool has_touch;

    std::shared_ptr<ActionStats> action_stats;

    static float sigmoid(float x);
    float compute_aim_angle(const std::unique_ptr<EnemyTankFactory> &other_tank);
};

#endif//ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H
