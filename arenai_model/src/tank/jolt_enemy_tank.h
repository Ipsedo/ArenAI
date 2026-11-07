//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_JOLT_ENEMY_TANK_H
#define ARENAI_JOLT_ENEMY_TANK_H

#include <memory>

#include <arenai_model/shell_economy.h>
#include <arenai_model/tank.h>

#include "./jolt_tank.h"

namespace arenai::model {

    struct TrackedShell {
        std::weak_ptr<ShellItem> shell;

        // closest approach over the whole trajectory, measured against the segment
        // travelled each frame (last_shell_pos → current position)
        glm::vec3 last_shell_pos;
        float min_distance;
        glm::vec3 enemy_pos_at_t;
        glm::vec3 shell_pos_at_t;
        bool has_sample;

        // last shell position, recorded at contact (the shell is already removed
        // from the engine when get_reward runs on the contact frame)
        glm::vec3 final_shell_pos;
        bool has_final_pos;
        // health fraction removed from the enemy part touched, 0 on a miss
        float damage_fraction;
        bool has_hit;
        bool has_killed;

        bool need_remove;
    };

    class JoltEnemyTank final : public JoltTank, public EnemyTank {
    public:
        JoltEnemyTank(
            JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::string &tank_prefix_name, glm::vec3 chassis_pos,
            float wanted_frame_frequency, const ShellEconomy &shell_economy);

        float get_reward() const override;
        int get_score() const override;

        bool is_dead() const override;
        bool is_first_frame_dead() const override;
        bool is_suicide() const override;

        bool consume_has_hit() override;
        bool consume_has_kill() override;
        bool consume_has_fire() override;

        void on_death() override;

        bool is_out_of_shells() const override;
        bool has_shells_in_flight() const override;
        void respawn(glm::vec3 chassis_pos) override;

        void tick(const std::vector<std::shared_ptr<EnemyTank>> &tanks) override;

        float
        get_aim_potential(const std::vector<std::shared_ptr<EnemyTank>> &tanks) const override;

        RewardDetail get_last_reward_detail() const override;

        std::vector<float> get_proprioception() const override;
        bool get_can_fire() const override;

        // Tank methods resolved via JoltTank
        using JoltTank::get_camera;
        using JoltTank::get_canon;
        using JoltTank::get_chassis;
        using JoltTank::get_controllers;
        using JoltTank::get_items;
        using JoltTank::load_shell_shapes;

    private:
        int max_frames_upside_down;
        int curr_frame_upside_down;

        float aim_tolerance_min;
        float aim_tolerance_angle;

        int initial_nb_shells;
        int nb_shells;
        int max_shells;
        int fire_cooldown_frames;
        int curr_cooldown_frame;

        int shells_recharged_per_hit;

        bool is_dead_already_triggered;

        bool has_hit;
        bool has_kill;
        bool has_fired;
        // lifetime counters behind get_score (the has_* flags above are
        // per-frame and consumed by the environment)
        int killed_nb;
        int hit_nb;

        // written by get_reward(), which stays const: the split is a read of the same
        // computation, not a second one
        mutable RewardDetail last_reward_detail;
        std::vector<TrackedShell> tracked_shells;

        // shells fired since the last get_reward(), reported there once
        mutable int nb_fired_to_pay;

        void on_shell_fired(const std::shared_ptr<ShellItem> &shell);
        void
        on_shell_contact(const ShellItem *shell, const ShellContactInfo &shell_info, Item *item);

        int get_nearest_enemy_index(
            const std::vector<std::shared_ptr<EnemyTank>> &tanks, const glm::vec3 &pos) const;

        // aim quality of a shell leaving fire_pos at fire_velocity: its trajectory is
        // integrated against the world (ray tests, own bodies and shells excluded)
        float predict_aim(
            const glm::vec3 &fire_pos, const glm::vec3 &fire_velocity,
            const std::vector<std::shared_ptr<EnemyTank>> &tanks) const;

        float compute_aim_quality(
            const glm::vec3 &fire_pos, const glm::vec3 &enemy_pos,
            const glm::vec3 &shell_pos) const;

        void update_closest_approach(
            TrackedShell &tracked, const glm::vec3 &shell_pos,
            const std::vector<std::shared_ptr<EnemyTank>> &tanks) const;
    };

}// namespace arenai::model

#endif//ARENAI_JOLT_ENEMY_TANK_H
