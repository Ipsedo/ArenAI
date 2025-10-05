//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_CONVEX_H
#define PHYVR_CONVEX_H

#include <android/asset_manager.h>

#include "glm/glm.hpp"
#include "poly.h"

class Convex : public Poly {
public:
  Convex(
    AAssetManager *mgr, std::string objFileName, glm::vec3 pos, glm::vec3 scale,
    glm::mat4 rotationMatrix, float mass);
};

#endif// PHYVR_CONVEX_H
