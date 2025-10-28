//
// Created by samuel on 20/10/2025.
//

#include "./player_tank_factory.h"

PlayerTankFactory::PlayerTankFactory(
    const std::shared_ptr<AbstractFileReader> &fileReader, const std::string &tankPrefixName,
    const glm::vec3 &chassisPos, const float wanted_frame_frequency)
    : TankFactory(fileReader, tankPrefixName, chassisPos, wanted_frame_frequency) {}

void PlayerTankFactory::on_fired_shell_contact(Item *item) {}
