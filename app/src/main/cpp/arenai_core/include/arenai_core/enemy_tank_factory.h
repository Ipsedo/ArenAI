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
        glm::vec3 chassis_pos, float wanted_frequency);
    float get_reward();

    bool is_dead() override;

    std::vector<std::shared_ptr<Item>> dead_and_get_items();

    std::vector<float> get_proprioception();

    std::shared_ptr<ActionStats> get_action_stats();

protected:
    void on_fired_shell_contact(Item *item) override;

private:
    float reward;
    int max_frames_upside_down;
    int curr_frame_upside_down;

    bool is_dead_already_triggered;

    std::shared_ptr<ActionStats> action_stats;

    int max_frames_without_hit;
    int nb_frames_since_last_hit;
};

#endif//ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H
