//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_TANK_FACTORY_H
#define ARENAI_TANK_FACTORY_H

#include <memory>
#include <string>

#include <glm/glm.hpp>

#include <arenai_utils/file_reader.h>

#include "./tank.h"

namespace arenai::model {

    class TankFactory {
    public:
        virtual ~TankFactory() = default;

        virtual std::unique_ptr<EnemyTank>
        make_enemy_tank(const std::string &tank_prefix_name, glm::vec3 chassis_pos) = 0;

        virtual std::unique_ptr<PlayerTank>
        make_player_tank(const std::string &tank_prefix_name, glm::vec3 chassis_pos) = 0;
    };

}// namespace arenai::model

#endif// ARENAI_TANK_FACTORY_H
