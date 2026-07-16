//
// Created by samuel on 19/03/2023.
//

#include <arenai_model/engine.h>
#include <arenai_model/tank_factory.h>

#include "./bullet_engine.h"
#include "./tank/bullet_tank_factory.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    std::unique_ptr<AbstractPhysicEngine> make_physic_engine(const float wanted_frequency) {
        return std::make_unique<BulletPhysicEngine>(wanted_frequency);
    }

}// namespace arenai::model
