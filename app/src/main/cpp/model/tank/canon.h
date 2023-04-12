//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_CANON_H
#define PHYVR_CANON_H

#include "../../controller/controller.h"
#include "../../view/camera.h"
#include "../items/convex.h"

class CanonItem : public ConvexItem, public Controller, public Camera {
public:
  CanonItem(std::string name, AAssetManager *mgr, glm::vec3 pos,
            glm::vec3 rel_pos, glm::vec3 scale, float mass,
            btRigidBody *turret);

  void on_input(const user_input &input) override;

  glm::vec3 pos() override;

  glm::vec3 look() override;

  glm::vec3 up() override;

  std::vector<btTypedConstraint *> get_constraints() override;

private:
  float angle;
  btHingeConstraint *hinge;
};

#endif // PHYVR_CANON_H
