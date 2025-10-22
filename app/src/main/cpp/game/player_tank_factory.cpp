//
// Created by samuel on 20/10/2025.
//

#include "./player_tank_factory.h"

PlayerTankFactory::PlayerTankFactory(
    const std::shared_ptr<AbstractFileReader> &fileReader, const std::string &tankPrefixName,
    const glm::vec3 &chassisPos)
    : TankFactory(fileReader, tankPrefixName, chassisPos) {}

void PlayerTankFactory::on_fired_shell_contact(Item *item) {}
