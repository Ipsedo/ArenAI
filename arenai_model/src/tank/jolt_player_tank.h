//
// Created by samuel on 21/03/2026.
//

#ifndef ARENAI_JOLT_PLAYER_TANK_H
#define ARENAI_JOLT_PLAYER_TANK_H

#include <arenai_model/tank.h>

#include "./jolt_tank.h"

namespace arenai::model {

    class JoltPlayerTank final : public JoltTank, public PlayerTank {
    public:
        JoltPlayerTank(
            JoltPhysicEngine &engine,
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::string &tank_prefix_name, const glm::vec3 &chassis_pos,
            float wanted_frame_frequency);

        int get_score() const override;

        // Tank methods resolved via JoltTank
        using JoltTank::get_camera;
        using JoltTank::get_canon;
        using JoltTank::get_chassis;
        using JoltTank::get_controllers;
        using JoltTank::get_items;
        using JoltTank::is_dead;
        using JoltTank::load_shell_shapes;

    private:
        void on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item);

        int killed_nb;
        int hit_nb;
    };

}// namespace arenai::model

#endif//ARENAI_JOLT_PLAYER_TANK_H
