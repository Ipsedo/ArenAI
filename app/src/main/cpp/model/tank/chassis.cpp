//
// Created by samuel on 02/04/2023.
//

#include "./chassis.h"

#include <glm/gtc/type_ptr.hpp>

ChassisItem::ChassisItem(AAssetManager *mgr, glm::vec3 position,
                         glm::vec3 scale, float mass)
    : ConvexItem("tank_chassis",
                 std::make_shared<ObjShape>(mgr, "obj/anubis_chassis.obj"),
                 position, scale, mass) {
  btVector3 center_of_mass = ConvexItem::get_body()->getCenterOfMassPosition();

  btTransform center_of_mass_tr;
  center_of_mass_tr.setIdentity();
  center_of_mass_tr.setOrigin(center_of_mass + btVector3(0, -2, 0));

  ConvexItem::get_body()->setCenterOfMassTransform(center_of_mass_tr);
}
