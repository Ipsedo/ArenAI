//
// Created by samuel on 30/05/18.
//

#ifndef PHYVR_CAMERA_H
#define PHYVR_CAMERA_H

#include "glm/glm.hpp"

class Camera {
public:
  virtual glm::vec3 camPos(bool VR) = 0;

  virtual glm::vec3 camLookAtVec(bool VR) = 0;

  virtual glm::vec3 camUpVec(bool VR) = 0;
};

#endif// PHYVR_CAMERA_H
