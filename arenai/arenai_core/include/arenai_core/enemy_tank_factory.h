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

    float get_reward();

    float get_potential_reward(
        const std::vector<std::unique_ptr<EnemyTankFactory>> &all_enemy_tank_factories);

    bool is_dead() override;

    std::vector<std::shared_ptr<Item>> dead_and_get_items();

    std::vector<float> get_proprioception();

protected:
    void on_fired_shell_contact(Item *item) override;

private:
    std::string tank_prefix_name;

    float reward;
    int max_frames_upside_down;
    int curr_frame_upside_down;

    bool is_dead_already_triggered;

    float min_distance_potential_reward;
    float max_distance_potential_reward;
    float aim_min_angle_potential_reward;
    float aim_max_angle_potential_reward;

    float compute_aim_angle(const std::unique_ptr<EnemyTankFactory> &other_tank);
    static float compute_value_range_reward(float value, float min_value, float max_value);
};

#endif//ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H
