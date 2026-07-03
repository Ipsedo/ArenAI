//
// Created by samuel on 01/07/2026.
//

#include "./bullet_tank_factory.h"

#include "./enemy_tank.h"
#include "./player_tank.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    BulletTankFactory::BulletTankFactory(
        BulletPhysicEngine &engine, const float wanted_frame_frequency)
        : engine(engine), wanted_frame_frequency(wanted_frame_frequency) {}

    std::unique_ptr<EnemyTank> BulletTankFactory::make_enemy_tank(
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::string &tank_prefix_name, glm::vec3 chassis_pos) {
        return std::make_unique<BulletEnemyTank>(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency);
    }

    std::unique_ptr<PlayerTank> BulletTankFactory::make_player_tank(
        const std::shared_ptr<utils::AbstractFileReader> &file_reader,
        const std::string &tank_prefix_name, glm::vec3 chassis_pos) {
        return std::make_unique<BulletPlayerTank>(
            engine, file_reader, tank_prefix_name, chassis_pos, wanted_frame_frequency);
    }

}// namespace arenai::model
