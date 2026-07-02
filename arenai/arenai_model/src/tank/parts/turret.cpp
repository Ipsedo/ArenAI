//
// Created by samuel on 02/04/2023.
//

#include "./turret.h"

#include <algorithm>
#include <memory>
#include <utility>

TurretItem::TurretItem(
    const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
    const glm::vec3 pos, const glm::vec3 rel_pos, const glm::vec3 scale, const float mass,
    btRigidBody *chassis)
    : LifeItem(5), ConvexItem(
                       prefix_name + "_turret",
                       std::make_shared<ObjShape>(
                           file_reader, std::filesystem::path("obj") / "anubis_turret.obj"),
                       pos, scale, mass),
      angle(0.f) {

    const auto chassis_pivot = btVector3(rel_pos.x, rel_pos.y, rel_pos.z);
    const auto turret_pivot = btVector3(0.f, 0, 0.f);
    const auto axis = btVector3(0.f, 1.f, 0.f);

    hinge = new btHingeConstraint(
        *chassis, *ConvexItem::get_body(), chassis_pivot, turret_pivot, axis, axis, true);
    // hinge->setLimit(0, 0);
}

void TurretItem::on_input(const user_input &input) {
    // angle is the hinge target in radians; input.right_joystick.x is a per-frame
    // delta already expressed in rad/frame by the controller handler.
    angle += -input.right_joystick.x;

    if (angle < -static_cast<float>(M_PI)) angle += 2.f * static_cast<float>(M_PI);
    else if (angle > static_cast<float>(M_PI)) angle -= 2.f * static_cast<float>(M_PI);
    angle = std::clamp(angle, -static_cast<float>(M_PI), static_cast<float>(M_PI));

    hinge->setLimit(angle, angle);
}

std::vector<btTypedConstraint *> TurretItem::get_constraints() {
    auto constraints = BulletItem::get_constraints();
    constraints.push_back(hinge);
    return constraints;
}
