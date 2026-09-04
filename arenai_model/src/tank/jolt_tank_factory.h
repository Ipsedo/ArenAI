//
// Created by samuel on 01/07/2026.
//

#ifndef ARENAI_JOLT_TANK_FACTORY_H
#define ARENAI_JOLT_TANK_FACTORY_H

#include <arenai_model/tank_factory.h>

namespace arenai::model {

    class JoltPhysicEngine;

    class JoltTankFactory final : public TankFactory {
    public:
        JoltTankFactory(JoltPhysicEngine &engine, float wanted_frame_frequency);

        std::unique_ptr<EnemyTank> make_enemy_tank(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::string &tank_prefix_name, glm::vec3 chassis_pos, bool apply_timeout,
            float max_episode_seconds) override;

        std::unique_ptr<PlayerTank> make_player_tank(
            const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader,
            const std::string &tank_prefix_name, glm::vec3 chassis_pos) override;

    private:
        JoltPhysicEngine &engine;
        float wanted_frame_frequency;
    };

}// namespace arenai::model

#endif// ARENAI_JOLT_TANK_FACTORY_H
