//
// Created by samuel on 02/04/2023.
//

#include "./canon.h"

#include <algorithm>

#include <glm/gtc/type_ptr.hpp>

using namespace arenai;
using namespace arenai::model;
using namespace arenai::controller;

namespace {

    glm::mat4 to_glm(const JPH::RMat44 &m) {
        glm::mat4 result;
        for (int c = 0; c < 4; c++) {
            const auto column = m.GetColumn4(c);
            result[c] = glm::vec4(column.GetX(), column.GetY(), column.GetZ(), column.GetW());
        }
        return result;
    }

}// namespace

namespace arenai::model {

    CanonItem::CanonItem(
        const std::string &prefix_name, JoltPhysicEngine &engine,
        const std::shared_ptr<utils::AbstractResourceFileReader> &file_reader, glm::vec3 pos,
        glm::vec3 rel_pos, glm::vec3 scale, float mass, JPH::Body *turret,
        const float wanted_frame_frequency,
        const std::function<void(glm::vec3, glm::vec3, Item *)> &on_contact,
        const std::function<void(const std::shared_ptr<ShellItem> &)> &on_shell_fired)
        : LifeItem(5), ConvexItem(
                           prefix_name + "_canon", engine,
                           std::make_shared<ObjShape>(
                               file_reader, std::filesystem::path("obj") / "anubis_canon.obj"),
                           pos, scale, mass),
          angle(0.f), file_reader(file_reader), will_fire(false), on_contact(on_contact),
          on_shell_fired(on_shell_fired), wanted_frame_frequency(wanted_frame_frequency) {

        JPH::HingeConstraintSettings settings;
        settings.mSpace = JPH::EConstraintSpace::LocalToBodyCOM;

        settings.mPoint1 = JPH::RVec3(rel_pos.x, rel_pos.y, rel_pos.z);
        settings.mHingeAxis1 = JPH::Vec3::sAxisX();
        settings.mNormalAxis1 = JPH::Vec3::sAxisY();

        settings.mPoint2 = JPH::RVec3::sZero();
        settings.mHingeAxis2 = JPH::Vec3::sAxisX();
        settings.mNormalAxis2 = JPH::Vec3::sAxisY();

        hinge =
            static_cast<JPH::HingeConstraint *>(settings.Create(*turret, *ConvexItem::get_body()));

        hinge->SetMotorState(JPH::EMotorState::Position);
        hinge->SetTargetAngle(angle);
    }

    std::vector<std::shared_ptr<JoltItem>> CanonItem::produce_jolt_items() {
        if (will_fire) {
            const glm::mat4 m_matrix = to_glm(ConvexItem::get_body()->GetWorldTransform());

            glm::vec4 shell_pos(0.f, 0.f, 10.f, 1.f);
            shell_pos = m_matrix * shell_pos;

            will_fire = false;

            auto shell_item = std::make_shared<ShellItem>(
                get_engine(), file_reader, glm::vec3(shell_pos), glm::toQuat(m_matrix),
                glm::vec3(0.2f), 20.f, wanted_frame_frequency, on_contact);

            glm::vec4 force_vec(0.f, 0.f, 1.f, 0.f);
            force_vec = m_matrix * force_vec;

            // Bullet applied a 1.5e5 central force integrated over exactly one
            // fixed timestep: same impulse, applied as a starting velocity
            const glm::vec3 velocity =
                glm::vec3(force_vec) * 1.5e5f * wanted_frame_frequency / 20.f;
            shell_item->get_body()->SetLinearVelocity(
                JPH::Vec3(velocity.x, velocity.y, velocity.z));

            on_shell_fired(shell_item);

            return {shell_item};
        }

        return {};
    }

    std::vector<std::shared_ptr<Item>> CanonItem::get_produced_items() {
        auto jolt_items = produce_jolt_items();
        return {jolt_items.begin(), jolt_items.end()};
    }

    void CanonItem::apply_input(const user_input &input) {
        angle += input.right_joystick.y * 0.4f;

        angle =
            std::clamp(angle, -0.2f * static_cast<float>(M_PI), 0.2f * static_cast<float>(M_PI));

        hinge->SetTargetAngle(angle);

        if (input.fire_button.pressed) will_fire = true;
    }

    glm::vec3 CanonItem::pos() {
        const glm::mat4 model_mat = to_glm(ConvexItem::get_body()->GetWorldTransform());

        return model_mat * glm::vec4(0, 4, -20, 1);
    }

    glm::vec3 CanonItem::look() {
        const glm::mat4 model_mat = to_glm(ConvexItem::get_body()->GetWorldTransform());

        return model_mat * glm::vec4(0, 0, 1, 1);
    }

    glm::vec3 CanonItem::up() {
        const glm::mat4 model_mat = to_glm(ConvexItem::get_body()->GetWorldTransform());

        return model_mat * glm::vec4(0, 1, 0, 0);
    }

    std::vector<JPH::Ref<JPH::TwoBodyConstraint>> CanonItem::get_constraints() {
        auto constraints = JoltItem::get_constraints();
        constraints.push_back(hinge.GetPtr());
        return constraints;
    }

}// namespace arenai::model
