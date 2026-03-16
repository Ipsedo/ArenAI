//
// Created by samuel on 16/03/2026.
//

#include "./player_tank_factory.h"

DesktopPlayerTankFactory::DesktopPlayerTankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    const glm::vec3 &chassis_pos, const float wanted_frame_frequency)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency) {}

void DesktopPlayerTankFactory::on_fired_shell_contact(Item *item) {}
