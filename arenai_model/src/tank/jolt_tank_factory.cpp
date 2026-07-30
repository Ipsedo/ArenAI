//
// Created by samuel on 01/07/2026.
//

#include "./jolt_tank_factory.h"

#include "./jolt_enemy_tank.h"
#include "./jolt_player_tank.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    JoltTankFactory::JoltTankFactory(JoltPhysicEngine &engine, const float wanted_frame_frequency)
        : engine(engine), wanted_frame_frequency(wanted_frame_frequency) {}

    std::unique_ptr<EnemyTank> JoltTankFactory::make_enemy_tank(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::string &tank_prefix_name, glm::vec3 chassis_pos) {
        return std::make_unique<JoltEnemyTank>(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency);
    }

    std::unique_ptr<PlayerTank> JoltTankFactory::make_player_tank(
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
        const std::string &tank_prefix_name, glm::vec3 chassis_pos) {
        return std::make_unique<JoltPlayerTank>(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency);
    }

}// namespace arenai::model
