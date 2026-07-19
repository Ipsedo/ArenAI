//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_BULLET_ENEMY_TANK_H
#define ARENAI_BULLET_ENEMY_TANK_H

#include <optional>

#include <arenai_model/action_stats.h>
#include <arenai_model/tank.h>

#include "./bullet_tank.h"

namespace arenai::model {

    struct shoot_info {
        glm::vec3 start_pos;
        glm::vec3 hit_pos;
        bool has_touch_enemy;
        bool enemy_killed;
    };

    class BulletEnemyTank final : public BulletTank, public EnemyTank {
    public:
        BulletEnemyTank(
            BulletPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::string &tank_prefix_name, glm::vec3 chassis_pos,
            float wanted_frame_frequency);

        float get_reward(const std::vector<std::shared_ptr<EnemyTank>> &tanks) override;
        float get_phi(const std::vector<std::shared_ptr<EnemyTank>> &tanks) override;

        bool is_dead() override;
        bool is_first_frame_dead() override;
        bool is_suicide() const override;

        bool has_hit_other_tank() override;

        void on_death() override;

        std::vector<float> get_proprioception() override;

        std::shared_ptr<ActionStats> get_action_stats() override;

        // Tank methods resolved via BulletTank
        using BulletTank::get_camera;
        using BulletTank::get_canon;
        using BulletTank::get_chassis;
        using BulletTank::get_controllers;
        using BulletTank::get_items;
        using BulletTank::load_shell_shapes;

    private:
        std::string tank_prefix_name;

        int max_frames_upside_down;
        int curr_frame_upside_down;

        float distance_scale;
        float impact_distance_scale;
        float angle_scale;
        float optimal_distance;

        float fire_cost;
        float miss_cost;

        bool is_dead_already_triggered;

        bool has_touch;
        std::optional<shoot_info> last_shoot_info;
        std::shared_ptr<ActionStats> action_stats;

        void on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item);

        float compute_aim_angle(const std::shared_ptr<EnemyTank> &other_tank);

        int get_nearest_enemy_index(
            const std::vector<std::shared_ptr<EnemyTank>> &tanks, const glm::vec3 &pos) const;

        float compute_hit_reward(
            const glm::vec3 &fire_pos, const glm::vec3 &best_enemy_pos,
            const glm::vec3 &hit_pos) const;

        float compute_shoot_reward(const std::vector<std::shared_ptr<EnemyTank>> &tanks);
    };

}// namespace arenai::model

#endif//ARENAI_BULLET_ENEMY_TANK_H
