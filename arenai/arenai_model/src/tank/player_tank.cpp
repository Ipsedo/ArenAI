//
// Created by samuel on 21/03/2026.
//

#include "./player_tank.h"

#include "../bullet_engine.h"

BulletPlayerTank::BulletPlayerTank(
    BulletPhysicEngine &engine, const std::shared_ptr<AbstractFileReader> &file_reader,
    const std::string &tank_prefix_name, const glm::vec3 &chassis_pos,
    const float wanted_frame_frequency)
    : BulletTank(
        engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency,
        [this](const ShellContactInfo &info, Item *item) { on_fired_shell_contact(info, item); }),
      killed_nb(0), hit_nb(0) {}

void BulletPlayerTank::on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item) {
    for (const auto &i: get_items())
        if (i->get_name() == item->get_name()) return;

    if (const auto &life_item = dynamic_cast<LifeItem *>(item); life_item) {
        if (life_item->is_dead() && !life_item->is_already_dead()) {
            killed_nb++;
        } else if (!life_item->is_dead()) {
            hit_nb++;
        }
    }
}

int BulletPlayerTank::get_score() const { return killed_nb * 10 + hit_nb; }
