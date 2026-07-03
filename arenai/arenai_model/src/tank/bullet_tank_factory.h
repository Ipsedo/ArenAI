//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_BULLET_TANK_FACTORY_H
#define ARENAI_BULLET_TANK_FACTORY_H

#include <arenai_model/tank_factory.h>

namespace arenai::model {

    class BulletPhysicEngine;

    class BulletTankFactory final : public TankFactory {
    public:
        BulletTankFactory(BulletPhysicEngine &engine, float wanted_frame_frequency);

        std::unique_ptr<EnemyTank> make_enemy_tank(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::string &tank_prefix_name, glm::vec3 chassis_pos) override;

        std::unique_ptr<PlayerTank> make_player_tank(
            const std::shared_ptr<utils::AbstractFileReader> &file_reader,
            const std::string &tank_prefix_name, glm::vec3 chassis_pos) override;

    private:
        BulletPhysicEngine &engine;
        float wanted_frame_frequency;
    };

}// namespace arenai::model

#endif// ARENAI_BULLET_TANK_FACTORY_H
