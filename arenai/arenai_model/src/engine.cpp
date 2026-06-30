//
// Created by samuel on 19/03/2023.
//

#include <arenai_model/engine.h>
#include <arenai_model/tank_factory.h>

#include "./bullet_engine.h"
#include "./tank/bullet_tank_factory.h"

std::unique_ptr<AbstractPhysicEngine> make_physic_engine(const float wanted_frequency) {
    return std::make_unique<BulletPhysicEngine>(wanted_frequency);
}

std::shared_ptr<TankFactory> make_tank_factory(
    AbstractPhysicEngine &engine, const std::shared_ptr<AbstractFileReader> &file_reader,
    const float wanted_frame_frequency) {
    auto &bullet_engine = static_cast<BulletPhysicEngine &>(engine);
    return std::make_shared<BulletTankFactory>(bullet_engine, file_reader, wanted_frame_frequency);
}
