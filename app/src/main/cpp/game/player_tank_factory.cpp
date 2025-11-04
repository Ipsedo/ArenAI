//
// Created by samuel on 20/10/2025.
//

#include "./player_tank_factory.h"

PlayerTankFactory::PlayerTankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        const glm::vec3 &chassis_pos, const float wanted_frame_frequency)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency) {}

void PlayerTankFactory::on_fired_shell_contact(Item *item) {}
