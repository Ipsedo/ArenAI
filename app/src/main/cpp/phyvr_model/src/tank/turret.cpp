//
// Created by samuel on 02/04/2023.
//

#include "./turret.h"

#include <memory>
#include <utility>

TurretItem::TurretItem(
    const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
    glm::vec3 pos, glm::vec3 rel_pos, glm::vec3 scale, float mass, btRigidBody *chassis)
    : ConvexItem(
        prefix_name + "_turret", std::make_shared<ObjShape>(file_reader, "obj/anubis_turret.obj"),
        pos, scale, mass),
      angle(0.f) {

    btVector3 chassis_pivot = btVector3(rel_pos.x, rel_pos.y, rel_pos.z);
    btVector3 turret_pivot = btVector3(0.f, 0, 0.f);
    btVector3 axis = btVector3(0.f, 1.f, 0.f);

    hinge = new btHingeConstraint(
        *chassis, *ConvexItem::get_body(), chassis_pivot, turret_pivot, axis, axis, true);
    hinge->setLimit(0, 0);
}

void TurretItem::on_input(const user_input &input) {
    angle += -input.right_joystick.x * 2.f;

    angle = angle > 1.f ? 1.f : angle;
    angle = angle < -1.f ? -1.f : angle;

    hinge->setLimit(angle * (float) M_PI * 0.6f, angle * (float) M_PI * 0.6f);
}

std::vector<btTypedConstraint *> TurretItem::get_constraints() {
    auto constraints = Item::get_constraints();
    constraints.push_back(hinge);
    return constraints;
}
