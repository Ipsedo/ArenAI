//
// Created by samuel on 02/04/2023.
//

#include "./wheel.h"

#include <Jolt/Physics/Constraints/SixDOFConstraint.h>

#include <arenai_model/constants.h>

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    using EAxis = JPH::SixDOFConstraintSettings::EAxis;

    WheelItem::WheelItem(
        const std::string &prefix_name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, const glm::vec3 pos,
        const glm::vec3 rel_pos, const glm::vec3 scale, const float mass, JPH::Body *chassis,
        float front_axle_z)
        : LifeItem(5), ConvexItem(
                           prefix_name + "_wheel", engine,
                           std::make_shared<ObjShape>(
                               file_reader, std::filesystem::path("obj") / "anubis_wheel.obj"),
                           pos, scale, mass) {

        JPH::SixDOFConstraintSettings settings;
        settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;

        settings.mPosition1 = JPH::RVec3(rel_pos.x, rel_pos.y, rel_pos.z);
        settings.mAxisX1 = JPH::Vec3::sAxisX();
        settings.mAxisY1 = JPH::Vec3::sAxisY();

        settings.mPosition2 = JPH::RVec3::sZero();
        settings.mAxisX2 = JPH::Vec3::sAxisX();
        settings.mAxisY2 = JPH::Vec3::sAxisY();

        settings.mSwingType = JPH::ESwingType::Pyramid;

        settings.MakeFixedAxis(EAxis::TranslationX);
        settings.SetLimitedAxis(EAxis::TranslationY, -0.4f, 0.f);
        settings.MakeFixedAxis(EAxis::TranslationZ);

        settings.MakeFreeAxis(EAxis::RotationX);
        settings.SetLimitedAxis(EAxis::RotationY, 0.f, 0.f);
        settings.MakeFixedAxis(EAxis::RotationZ);

        settings.mMotorSettings[EAxis::TranslationY].mSpringSettings =
            JPH::SpringSettings(JPH::ESpringMode::StiffnessAndDamping, 2e5f, 30.f);

        settings.mMotorSettings[EAxis::RotationX].mMinTorqueLimit = -2e4f;
        settings.mMotorSettings[EAxis::RotationX].mMaxTorqueLimit = 2e4f;

        auto *constraint = settings.Create(*chassis, *ConvexItem::get_body());

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        hinge = static_cast<JPH::SixDOFConstraint *>(constraint);

        hinge->SetMotorState(EAxis::TranslationY, JPH::EMotorState::Position);
        hinge->SetTargetPositionCS(JPH::Vec3(0.f, -0.2f, 0.f));

        hinge->SetMotorState(EAxis::RotationX, JPH::EMotorState::Velocity);
        hinge->SetTargetAngularVelocityCS(JPH::Vec3::sZero());

        ConvexItem::get_body()->SetFriction(1.f);

        // for differential
        wheel_center_pos_rel_to_chassis = rel_pos;
        this->front_axle_z = front_axle_z;
    }

    void WheelItem::apply_input(const controller::user_input &input) {
        const auto radial_velocity = -input.left_joystick.y * WHEEL_RADIAL_VELOCITY;

        const float angle = input.left_joystick.x * WHEEL_DIRECTION_MAX_RADIAN;

        hinge->SetTargetAngularVelocityCS(
            JPH::Vec3(-adjust_rotation_velocity_differential(angle, radial_velocity), 0.f, 0.f));
    }

    std::vector<JPH::Ref<JPH::TwoBodyConstraint>> WheelItem::get_constraints() {
        auto constraints = JoltItem::get_constraints();
        constraints.emplace_back(hinge.GetPtr());
        return constraints;
    }

    float WheelItem::adjust_rotation_velocity_differential(
        const float front_wheel_orientation_radian, const float original_rotation_velocity) const {

        const float delta = front_wheel_orientation_radian;

        if (constexpr float eps = 1e-6f;
            std::fabs(delta) < eps || std::fabs(std::tan(delta)) < eps) {
            return original_rotation_velocity;
        }

        const float Rc = -2.f * front_axle_z / std::tan(delta);

        const auto xw = wheel_center_pos_rel_to_chassis.x;
        const auto zw = wheel_center_pos_rel_to_chassis.z;

        const float rw =
            std::sqrt((xw - Rc) * (xw - Rc) + (zw + front_axle_z) * (zw + front_axle_z));

        const float rc = std::sqrt(Rc * Rc + front_axle_z * front_axle_z);

        return original_rotation_velocity * rw / rc;
    }

    /*
 * DirectionalWheelItem
 */

    void DirectionalWheelItem::apply_input(const controller::user_input &input) {
        WheelItem::apply_input(input);

        const float angle = input.left_joystick.x * WHEEL_DIRECTION_MAX_RADIAN * angle_factor;

        hinge->SetRotationLimits(
            JPH::Vec3(-JPH::JPH_PI, -angle, 0.f), JPH::Vec3(JPH::JPH_PI, -angle, 0.f));
    }

    DirectionalWheelItem::DirectionalWheelItem(
        const std::string &name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, const glm::vec3 pos,
        const glm::vec3 rel_pos, const glm::vec3 scale, const float mass, JPH::Body *chassis,
        float front_axle_z, const float angle_factor)
        : WheelItem(name, engine, file_reader, pos, rel_pos, scale, mass, chassis, front_axle_z),
          angle_factor(angle_factor) {}

}// namespace arenai::model
