//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_SPHERE_H
#define PHYVR_SPHERE_H

#include "../../utils/assets.h"
#include "../poly.h"
#include "glm/glm.hpp"
#include <android/asset_manager.h>

class Sphere : public Poly {
public:
  Sphere(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat,
         float mass);

  Sphere(GLDrawable *drawable, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat,
         float mass);
};

#endif // PHYVR_SPHERE_H
