//
// Created by samuel on 19/03/2023.
//

#include <arenai_model/engine.h>

#include "bullet_engine.h"

std::unique_ptr<AbstractPhysicEngine> make_physic_engine(const float wanted_frequency) {
    return std::make_unique<BulletPhysicEngine>(wanted_frequency);
}
