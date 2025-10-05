//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_CONE_H
#define PHYVR_CONE_H

#include <android/asset_manager.h>

#include "../../utils/assets.h"
#include "../poly.h"
#include "glm/glm.hpp"

class Cone : public Poly {
public:
    Cone(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);

    Cone(DiffuseModel *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);
};

#endif// PHYVR_CONE_H
