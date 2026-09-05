//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_JOLT_ENEMY_TANK_H
#define ARENAI_JOLT_ENEMY_TANK_H

#include <memory>

#include <arenai_model/tank.h>

#include "./jolt_tank.h"

namespace arenai::model {

    struct TrackedShell {
        std::weak_ptr<ShellItem> shell;
        glm::vec3 fire_pos;

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
            float wanted_frame_frequency, bool apply_timeout, float max_episode_seconds);

        float get_reward() const override;

        bool is_dead() const override;
        bool is_first_frame_dead() const override;
        bool is_suicide() const override;
        bool is_timeout() const override;

        bool consume_has_hit() override;
        bool consume_has_kill() override;
        bool consume_has_fire() override;

        void on_death() override;

        void tick(const std::vector<std::shared_ptr<EnemyTank>> &tanks) override;

        RewardDetail get_last_reward_detail() const override;

        std::vector<float> get_proprioception() const override;

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

        float miss_distance_scale;
        float miss_distance_exponent;
        float hit_reward_scale;
        float aim_quality_baseline;

        float hit_received_cost;

        int initial_nb_shells;
        int nb_shells;
        int max_shells;
        int fire_cooldown_frames;
        int curr_cooldown_frame;

        int shells_recharged_per_hit;

        int nb_frames_per_shell_regen;
        int curr_frame_shell_regen;

        bool is_dead_already_triggered;

        bool apply_timeout;
        int max_frames_without_hit;
        int remaining_frames;
        int nb_frames_added_when_hit;
        int nb_frames_added_when_kill;
        int max_episode_frames;

        bool has_hit;
        bool has_kill;
        bool has_fired;

        // written by get_reward(), which stays const: the split is a read of the same
        // computation, not a second one
        mutable RewardDetail last_reward_detail;
        std::vector<TrackedShell> tracked_shells;

        void on_shell_fired(const std::shared_ptr<ShellItem> &shell);
        void
        on_shell_contact(const ShellItem *shell, const ShellContactInfo &shell_info, Item *item);

        int get_nearest_enemy_index(
            const std::vector<std::shared_ptr<EnemyTank>> &tanks, const glm::vec3 &pos) const;

        void update_closest_approach(
            TrackedShell &tracked, const glm::vec3 &shell_pos,
            const std::vector<std::shared_ptr<EnemyTank>> &tanks) const;

        float compute_hit_reward(
            const glm::vec3 &fire_pos, const glm::vec3 &enemy_pos,
            const glm::vec3 &shell_pos) const;
    };

}// namespace arenai::model

#endif//ARENAI_JOLT_ENEMY_TANK_H
