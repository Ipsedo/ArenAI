//
// Created by samuel on 16/03/2026.
//

#ifndef ARENAI_DESKTOP_PLAYER_TANK_FACTORY_H
#define ARENAI_DESKTOP_PLAYER_TANK_FACTORY_H

#include <arenai_model/tank_factory.h>

class DesktopPlayerTankFactory : public TankFactory {
public:
    DesktopPlayerTankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        const glm::vec3 &chassis_pos, float wanted_frame_frequency);

protected:
    void on_fired_shell_contact(Item *item) override;
};

#endif//ARENAI_DESKTOP_PLAYER_TANK_FACTORY_H
