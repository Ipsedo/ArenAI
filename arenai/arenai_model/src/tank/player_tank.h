//
// Created by samuel on 21/03/2026.
//

#ifndef ARENAI_BULLET_PLAYER_TANK_H
#define ARENAI_BULLET_PLAYER_TANK_H

#include <arenai_model/tank.h>

#include "./bullet_tank.h"

namespace arenai::model {

    class BulletPlayerTank final : public BulletTank, public PlayerTank {
    public:
        BulletPlayerTank(
            BulletPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::string &tank_prefix_name, const glm::vec3 &chassis_pos,
            float wanted_frame_frequency);

        int get_score() const override;

        // Tank methods resolved via BulletTank
        using BulletTank::get_camera;
        using BulletTank::get_canon;
        using BulletTank::get_chassis;
        using BulletTank::get_controllers;
        using BulletTank::get_items;
        using BulletTank::is_dead;
        using BulletTank::load_shell_shapes;

    private:
        void on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item);

        int killed_nb;
        int hit_nb;
    };

}// namespace arenai::model

#endif//ARENAI_BULLET_PLAYER_TANK_H
