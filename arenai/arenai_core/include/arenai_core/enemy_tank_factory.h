//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H
#define ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H

#include <optional>

#include <arenai_model/tank_factory.h>

struct shoot_info {
    glm::vec3 start_pos;
    glm::vec3 hit_pos;
    bool has_touch_enemy;
    bool enemy_killed;
};

class EnemyTankFactory final : public TankFactory {
public:
    EnemyTankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        glm::vec3 chassis_pos, float wanted_frame_frequency);

    float get_reward(const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories);

    bool is_dead() override;
    bool is_suicide() const;

    bool has_hit_other_tank();

    std::vector<std::shared_ptr<Item>> dead_and_get_items();

    std::vector<float> get_proprioception();

protected:
    void on_fired_shell_contact(ShellItem *shell, Item *item) override;

private:
    std::string tank_prefix_name;

    int max_frames_upside_down;
    int curr_frame_upside_down;

    float distance_scale;

    bool is_dead_already_triggered;

    bool has_touch;

    std::optional<shoot_info> last_shoot_info;

    float compute_aim_angle(const std::unique_ptr<EnemyTankFactory> &other_tank);

    std::tuple<int, float>
    get_best_score(const std::vector<std::unique_ptr<EnemyTankFactory>> &tank_factories);

    static float compute_shoot_reward(
        const glm::vec3 &fire_pos, const glm::vec3 &best_enemy_pos, const glm::vec3 &hit_pos);
};

#endif//ARENAI_TRAIN_HOST_ENEMY_TANK_FACTORY_H
