//
// Created by samuel on 19/03/2023.
//

#include <arenai_model/engine.h>
#include <arenai_model/tank_factory.h>

#include "./jolt_engine.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    std::unique_ptr<AbstractPhysicEngine> make_physic_engine(const float wanted_frequency) {
        return std::make_unique<JoltPhysicEngine>(wanted_frequency);
    }

}// namespace arenai::model
