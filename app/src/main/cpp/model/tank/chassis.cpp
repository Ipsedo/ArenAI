//
// Created by samuel on 02/04/2023.
//

#include "./chassis.h"

#include <glm/gtc/type_ptr.hpp>

ChassisItem::ChassisItem(AAssetManager *mgr, glm::vec3 position,
                         glm::vec3 scale, float mass)
    : ConvexItem("tank_chassis",
                 std::make_shared<ObjShape>(mgr, "obj/anubis_chassis.obj"),
                 position, scale, mass) {}
