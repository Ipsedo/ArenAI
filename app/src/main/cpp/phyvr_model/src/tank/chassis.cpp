//
// Created by samuel on 02/04/2023.
//

#include "./chassis.h"

#include <glm/gtc/type_ptr.hpp>

ChassisItem::ChassisItem(
    const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
    glm::vec3 position, glm::vec3 scale, float mass)
    : LifeItem(60), ConvexItem(
                        prefix_name + "_chassis",
                        std::make_shared<ObjShape>(file_reader, "obj/anubis_chassis.obj"), position,
                        scale, mass) {
    btVector3 center_of_mass = ConvexItem::get_body()->getCenterOfMassPosition();

    btTransform center_of_mass_tr;
    center_of_mass_tr.setIdentity();
    center_of_mass_tr.setOrigin(center_of_mass + btVector3(0, -2, 0));

    ConvexItem::get_body()->setCenterOfMassTransform(center_of_mass_tr);
}
