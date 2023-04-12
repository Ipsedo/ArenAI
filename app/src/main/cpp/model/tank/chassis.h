//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_CHASSIS_H
#define PHYVR_CHASSIS_H

#include "../../view/camera.h"
#include "../items/convex.h"

class ChassisItem : public ConvexItem {
public:
  ChassisItem(AAssetManager *mgr, glm::vec3 position, glm::vec3 scale,
              float mass);
};

#endif // PHYVR_CHASSIS_H
