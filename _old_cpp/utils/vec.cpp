//
// Created by samuel on 16/08/18.
//

#include "vec.h"

glm::vec3 btVector3ToVec3(btVector3 v) {
  return glm::vec3(v.x(), v.y(), v.z());
}
