//
// Created by samuel on 02/04/2023.
//

#include "./turret.h"

#include <algorithm>
#include <memory>
#include <utility>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::controller;

namespace arenai::model {

    TurretItem::TurretItem(
        const std::string &prefix_name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, const glm::vec3 pos,
        const glm::vec3 rel_pos, const glm::vec3 scale, const float mass, JPH::Body *chassis)
        : LifeItem(5), ConvexItem(
                           prefix_name + "_turret", engine,
                           std::make_shared<ObjShape>(
                               file_reader, std::filesystem::path("obj") / "anubis_turret.obj"),
                           pos, scale, mass),
          angle(0.f) {

        JPH::HingeConstraintSettings settings;
        settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;

        settings.mPoint1 = JPH::RVec3(rel_pos.x, rel_pos.y, rel_pos.z);
        settings.mHingeAxis1 = JPH::Vec3::sAxisY();
        settings.mNormalAxis1 = JPH::Vec3::sAxisX();

        settings.mPoint2 = JPH::RVec3::sZero();
        settings.mHingeAxis2 = JPH::Vec3::sAxisY();
        settings.mNormalAxis2 = JPH::Vec3::sAxisX();

        hinge =
            static_cast<JPH::HingeConstraint *>(settings.Create(*chassis, *ConvexItem::get_body()));
        // like Bullet's limit-less hinge: free until the first input engages
        // the servo
    }

    void TurretItem::apply_input(const user_input &input) {
        // angle is the hinge target in radians; input.right_joystick.x is a per-frame
        // delta already expressed in rad/frame by the controller handler.
        angle += -input.right_joystick.x;

        if (angle < -static_cast<float>(M_PI)) angle += 2.f * static_cast<float>(M_PI);
        else if (angle > static_cast<float>(M_PI)) angle -= 2.f * static_cast<float>(M_PI);
        angle = std::clamp(angle, -static_cast<float>(M_PI), static_cast<float>(M_PI));

        hinge->SetMotorState(JPH::EMotorState::Position);
        hinge->SetTargetAngle(angle);
    }

    std::vector<JPH::Ref<JPH::TwoBodyConstraint>> TurretItem::get_constraints() {
        auto constraints = JoltItem::get_constraints();
        constraints.push_back(hinge.GetPtr());
        return constraints;
    }

}// namespace arenai::model
