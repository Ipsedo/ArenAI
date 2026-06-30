//
// Created by samuel on 21/03/2026.
//

#ifndef ARENAI_DESKTOP_PLAYER_TANK_FACTORY_H
#define ARENAI_DESKTOP_PLAYER_TANK_FACTORY_H

#include <arenai_model/tank_factory.h>

class PlayerTankFactory : public TankFactory {
public:
    PlayerTankFactory(
        const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
        const glm::vec3 &chassis_pos, float wanted_frame_frequency);

    int get_score() const;

protected:
    void on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item) override;

private:
    int killed_nb;
    int hit_nb;
};

#endif//ARENAI_DESKTOP_PLAYER_TANK_FACTORY_H
