//
// Created by samuel on 02/04/2023.
//

#include "./wheel.h"

#include <utility>

WheelItem::WheelItem(
    const std::string &prefix_name, const std::shared_ptr<AbstractFileReader> &file_reader,
    const glm::vec3 pos, const glm::vec3 rel_pos, const glm::vec3 scale, const float mass,
    btRigidBody *chassis, float front_axle_z)
    : LifeItem(5),
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
    for (int axis_to_disable[] = {0, 2, 5}; const auto axis: axis_to_disable) {
        hinge->setParam(BT_CONSTRAINT_STOP_ERP, 0.9, axis);
        hinge->setParam(BT_CONSTRAINT_STOP_CFM, 0.0, axis);
        hinge->setLimit(axis, 0, 0);
        hinge->enableMotor(axis, false);
        hinge->enableSpring(axis, false);
    }

    ConvexItem::get_body()->setFriction(500.f);

    // for differential
    wheel_center_pos_rel_to_chassis = rel_pos;
    this->front_axle_z = front_axle_z;
}

void WheelItem::on_input(const user_input &input) {
    constexpr int motor_axis = 3;
    const auto radial_velocity = -input.left_joystick.y * static_cast<float>(M_PI) * 5.f;

    const float angle = input.left_joystick.x * WHEEL_DIRECTION_MAX_RADIAN;

    hinge->setTargetVelocity(
        motor_axis, adjust_rotation_velocity_differential(angle, radial_velocity));
}

std::vector<btTypedConstraint *> WheelItem::get_constraints() {
    auto constraints = Item::get_constraints();
    constraints.push_back(hinge);
    return constraints;
}

float WheelItem::adjust_rotation_velocity_differential(
    const float front_wheel_orientation_radian, const float original_rotation_velocity) const {

    const float delta = front_wheel_orientation_radian;

    constexpr float eps = 1e-6f;
    if (std::fabs(delta) < 1e-6f || std::fabs(std::tan(delta)) < eps) {
        return original_rotation_velocity;
    }

    const float Rc = -front_axle_z / std::tan(delta);

    const auto xw = wheel_center_pos_rel_to_chassis.x;
    const auto zw = wheel_center_pos_rel_to_chassis.z;

    const float rw = std::sqrt((xw - Rc) * (xw - Rc) + zw * zw);
    const float rc = std::fabs(Rc);

    if (rc < eps) {
        const float ratio = rw / eps;
        return original_rotation_velocity * ratio;
    }

    const float ratio = rw / rc;
    return original_rotation_velocity * ratio;
}

/*
 * DirectionalWheelItem
 */

void DirectionalWheelItem::on_input(const user_input &input) {
    WheelItem::on_input(input);

    constexpr int motor_axis = 4;
    const float angle = input.left_joystick.x * WHEEL_DIRECTION_MAX_RADIAN;

    hinge->setLimit(motor_axis, angle, angle);
}

DirectionalWheelItem::DirectionalWheelItem(
    const std::string &name, const std::shared_ptr<AbstractFileReader> &file_reader,
    const glm::vec3 pos, const glm::vec3 rel_pos, const glm::vec3 scale, const float mass,
    btRigidBody *chassis, float front_axle_z)
    : WheelItem(name, file_reader, pos, rel_pos, scale, mass, chassis, front_axle_z) {}
