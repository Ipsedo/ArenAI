//
// Created by samuel on 13/04/2023.
//

#ifndef PHYVR_AMMU_H
#define PHYVR_AMMU_H

#include "../items/convex.h"

#include <glm/gtx/quaternion.hpp>

class ShellItem : public ConvexItem {
public:
  static std::shared_ptr<Shape> load_shape(AAssetManager *mgr);

  ShellItem(AAssetManager *mgr, glm::vec3 pos, glm::quat rot, glm::vec3 scale,
            float mass);

  inline const static std::string NAME = "shell_item";
};

#endif // PHYVR_AMMU_H
