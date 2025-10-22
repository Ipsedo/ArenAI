//
// Created by samuel on 20/10/2025.
//

#ifndef PHYVR_PLAYER_TANK_FACTORY_H
#define PHYVR_PLAYER_TANK_FACTORY_H

#include <arenai_model/tank_factory.h>

class PlayerTankFactory final : public TankFactory {
public:
    PlayerTankFactory(
        const std::shared_ptr<AbstractFileReader> &fileReader, const std::string &tankPrefixName,
        const glm::vec3 &chassisPos);

protected:
    void on_fired_shell_contact(Item *item) override;
};

#endif//PHYVR_PLAYER_TANK_FACTORY_H
