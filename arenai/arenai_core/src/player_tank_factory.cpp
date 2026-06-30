//
// Created by samuel on 21/03/2026.
//

#include <arenai_core/player_tank_factory.h>

PlayerTankFactory::PlayerTankFactory(
    const std::shared_ptr<AbstractFileReader> &file_reader, const std::string &tank_prefix_name,
    const glm::vec3 &chassis_pos, const float wanted_frame_frequency)
    : TankFactory(file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency), killed_nb(0),
      hit_nb(0) {}

void PlayerTankFactory::on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item) {
    for (const auto &i: get_items())
        if (i->get_name() == item->get_name()) return;// self shoot

    if (const auto &life_item = dynamic_cast<LifeItem *>(item); life_item) {
        if (life_item->is_dead() && !life_item->is_already_dead()) {
            killed_nb++;
        } else if (!life_item->is_dead()) {
            hit_nb++;
        }
    }
}

int PlayerTankFactory::get_score() const {
    // TODO real score
    return killed_nb * 10 + hit_nb;
}
