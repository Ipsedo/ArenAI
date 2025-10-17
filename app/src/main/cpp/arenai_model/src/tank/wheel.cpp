//
// Created by samuel on 02/04/2023.
//

#include "./wheel.h"

#include <utility>

WheelItem::WheelItem(
    const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
    const glm::vec3 pos, const glm::vec3 rel_pos, const glm::vec3 scale, const float mass,
    btRigidBody *chassis)
    : LifeItem(10),
      ConvexItem(
          prefix_name + "_wheel", std::make_shared<ObjShape>(file_reader, "obj/anubis_wheel.obj"),
          pos, scale, mass) {

    btTransform frame_in_wheel;
    frame_in_wheel.setIdentity();
    frame_in_wheel.setOrigin(btVector3(0, 0, 0));

    btTransform frame_in_chassis;
    frame_in_chassis.setIdentity();
    frame_in_chassis.setOrigin(btVector3(rel_pos.x, rel_pos.y, rel_pos.z));

    hinge = new btGeneric6DofSpring2Constraint(
        *chassis, *ConvexItem::get_body(), frame_in_chassis, frame_in_wheel, RO_XYZ);

    hinge->setAngularLowerLimit(btVector3(1, 0, 0));
    hinge->setAngularUpperLimit(btVector3(-1, 0, 0));

    // Linear limits
    hinge->setLinearLowerLimit(btVector3(0, -0.4f, 0));
    hinge->setLinearUpperLimit(btVector3(0, 0, 0));

    constexpr int motor_axis = 3;
    hinge->enableMotor(motor_axis, true);
    hinge->setMaxMotorForce(motor_axis, 4e3f);
    hinge->setTargetVelocity(motor_axis, 0.f);

    constexpr int index = 1;
    hinge->enableSpring(index, true);
    hinge->setDamping(index, 30.f, true);
    hinge->setStiffness(index, 100.f, true);
    hinge->setBounce(index, 1e-2f);
    hinge->setEquilibriumPoint(index, -0.2f);

    // disable axis
    for (float axis_to_disable[] = {0, 2, 5}; const auto axis: axis_to_disable) {
        hinge->setParam(BT_CONSTRAINT_STOP_ERP, 0.9, axis);
        hinge->setParam(BT_CONSTRAINT_STOP_CFM, 0.0, axis);
        hinge->setLimit(axis, 0, 0);
        hinge->enableMotor(axis, false);
        hinge->enableSpring(axis, false);
    }

    ConvexItem::get_body()->setFriction(500.f);
}

void WheelItem::on_input(const user_input &input) {
    constexpr int motor_axis = 3;
    const auto radial_velocity = -input.left_joystick.y * static_cast<float>(M_PI) * 5.f;

    hinge->setTargetVelocity(motor_axis, radial_velocity);
}

std::vector<btTypedConstraint *> WheelItem::get_constraints() {
    auto constraints = Item::get_constraints();
    constraints.push_back(hinge);
    return constraints;
}

/*
 * DirectionalWheelItem
 */

void DirectionalWheelItem::on_input(const user_input &input) {
    WheelItem::on_input(input);

    constexpr int motor_axis = 4;
    const float angle = input.left_joystick.x * static_cast<float>(M_PI) / 8.f;

    hinge->setLimit(motor_axis, angle, angle);
}

DirectionalWheelItem::DirectionalWheelItem(
    std::string name, const std::shared_ptr<AbstractFileReader> &file_reader, const glm::vec3 pos,
    const glm::vec3 rel_pos, const glm::vec3 scale, const float mass, btRigidBody *chassis)
    : WheelItem(std::move(name), file_reader, pos, rel_pos, scale, mass, chassis) {}
