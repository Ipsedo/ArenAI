//
// Created by samuel on 26/05/18.
//

#ifndef PHYVR_BOX_H
#define PHYVR_BOX_H

#include "../../graphics/drawable/modelvbo.h"
#include "../../utils/assets.h"
#include "../poly.h"
#include "btBulletDynamicsCommon.h"
#include "glm/glm.hpp"
#include <android/asset_manager.h>

class Box : public Poly {
public:
  Box(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat,
      float mass);
};

#endif // PHYVR_CUBE_H
