//
// Created by samuel on 02/04/2023.
//

#include "./turret.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numbers>

using namespace arenai;
using namespace arenai::model;

namespace {

    constexpr float PI = std::numbers::pi_v<float>;
    constexpr float TWO_PI = 2.f * PI;

}// namespace

namespace arenai::model {

    TurretItem::TurretItem(
        const std::string &prefix_name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, const glm::vec3 pos,
        const glm::vec3 rel_pos, const glm::vec3 scale, const float mass, JPH::Body *chassis,
        const float max_rad_per_frame)
        : LifeItem(5), ConvexItem(
                           prefix_name + "_turret", engine,
                           std::make_shared<ObjShape>(
                               file_reader, std::filesystem::path("obj") / "anubis_turret.obj"),
                           pos, scale, mass),
          angle(0.f), max_rad_per_frame(max_rad_per_frame) {

        JPH::HingeConstraintSettings settings;
        settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;

        settings.mPoint1 = JPH::RVec3(rel_pos.x, rel_pos.y, rel_pos.z);
        settings.mHingeAxis1 = JPH::Vec3::sAxisY();
        settings.mNormalAxis1 = JPH::Vec3::sAxisX();

        settings.mPoint2 = JPH::RVec3::sZero();
        settings.mHingeAxis2 = JPH::Vec3::sAxisY();
        settings.mNormalAxis2 = JPH::Vec3::sAxisX();

        auto *constraint = settings.Create(*chassis, *ConvexItem::get_body());

        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-static-cast-downcast)
        hinge = static_cast<JPH::HingeConstraint *>(constraint);
    }

    void TurretItem::apply_input(const controller::user_input &input) {
        const float target = input.right_joystick.x * PI;

        // std::remainder keeps the error in [-pi, pi]: the turret always takes the shortest
        // way around, so crossing the back stays a small move
        const float delta = std::clamp(
            std::remainder(target - angle, TWO_PI), -max_rad_per_frame, max_rad_per_frame);

        angle = std::remainder(angle + delta, TWO_PI);

        hinge->SetMotorState(JPH::EMotorState::Position);
        hinge->SetTargetAngle(angle);
    }

    float TurretItem::get_angle() const { return angle; }

    std::vector<JPH::Ref<JPH::TwoBodyConstraint>> TurretItem::get_constraints() {
        auto constraints = JoltItem::get_constraints();
        constraints.emplace_back(hinge.GetPtr());
        return constraints;
    }

}// namespace arenai::model
