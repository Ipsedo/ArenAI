//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_CHASSIS_H
#define PHYVR_CHASSIS_H

#include "../../view/camera.h"
#include "../items/convex.h"

class ChassisItem : public ConvexItem, public Camera {
public:
  ChassisItem(AAssetManager *mgr, glm::vec3 position, glm::vec3 scale,
              float mass);

  glm::vec3 pos() override;

  glm::vec3 look() override;

  glm::vec3 up() override;

private:
  glm::vec3 get_pos_();
};

#endif // PHYVR_CHASSIS_H
