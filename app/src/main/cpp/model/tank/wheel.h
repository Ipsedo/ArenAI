//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_WHEEL_H
#define PHYVR_WHEEL_H

#include "../../controller/controller.h"
#include "../items/convex.h"

class WheelItem : public ConvexItem, public Controller {
public:
  WheelItem(std::string name, AAssetManager *mgr, glm::vec3 pos,
            glm::vec3 rel_pos, glm::vec3 scale, float mass,
            btRigidBody *chassis);
  void on_input(const user_input &input) override;

  std::vector<btTypedConstraint *> get_constraints() override;

protected:
  btGeneric6DofSpring2Constraint *hinge;
};

class DirectionalWheelItem : public WheelItem {
public:
  DirectionalWheelItem(std::string name, AAssetManager *mgr, glm::vec3 pos,
                       glm::vec3 rel_pos, glm::vec3 scale, float mass,
                       btRigidBody *chassis);
  void on_input(const user_input &input) override;
};

#endif // PHYVR_WHEEL_H
