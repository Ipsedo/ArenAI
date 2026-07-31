//
// Created by samuel on 21/03/2026.
//

#include "./jolt_player_tank.h"

#include "../jolt_engine.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    JoltPlayerTank::JoltPlayerTank(
        JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::string &tank_prefix_name, const glm::vec3 &chassis_pos,
        const float wanted_frame_frequency)
        : JoltTank(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency,
            [this](const ShellContactInfo &info, Item *item) {
                on_fired_shell_contact(info, item);
            }),
          killed_nb(0), hit_nb(0) {}

    void JoltPlayerTank::on_fired_shell_contact(const ShellContactInfo &shell_info, Item *item) {
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

    int JoltPlayerTank::get_score() const { return killed_nb * 10 + hit_nb; }

    void JoltPlayerTank::destroy() { remove_constraints_from_engine(); }

}// namespace arenai::model
