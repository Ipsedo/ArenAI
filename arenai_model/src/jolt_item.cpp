//
// Created by claude on 30/07/2026.
//

#include "./jolt_item.h"

#include <utility>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "./jolt_engine.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    JoltItem::JoltItem(std::string name, JoltPhysicEngine &engine)
        : Item(std::move(name)), engine(engine) {}

    glm::mat4 JoltItem::get_model_matrix() {
        const auto *body = get_body();

        JPH::RVec3 position = body->GetPosition();
        JPH::Quat rotation = body->GetRotation();

        if (body->IsDynamic() && body->IsActive()) {
            const float dt = engine.get_interpolation_delta();

            const JPH::Vec3 linear_velocity = body->GetLinearVelocity();
            const JPH::Vec3 angular_velocity = body->GetAngularVelocity();

            position += linear_velocity * dt;

            if (const float angle = angular_velocity.Length() * dt; std::abs(angle) > 1e-9f)
                rotation = (JPH::Quat::sRotation(angular_velocity.Normalized(), angle) * rotation)
                               .Normalized();
        }

        const glm::quat q(rotation.GetW(), rotation.GetX(), rotation.GetY(), rotation.GetZ());

        glm::mat4 matrix = glm::mat4_cast(q);
        matrix[3] = glm::vec4(position.GetX(), position.GetY(), position.GetZ(), 1.f);

        return matrix * glm::scale(glm::mat4(1.f), _get_scale());
    }

    glm::vec3 JoltItem::get_linear_velocity() {
        const auto *body = get_body();
        if (!body->IsDynamic()) return glm::vec3(0.f);

        const auto vel = body->GetLinearVelocity();
        return {vel.GetX(), vel.GetY(), vel.GetZ()};
    }

    glm::vec3 JoltItem::get_angular_velocity() {
        const auto *body = get_body();
        if (!body->IsDynamic()) return glm::vec3(0.f);

        const auto vel = body->GetAngularVelocity();
        return {vel.GetX(), vel.GetY(), vel.GetZ()};
    }

    std::vector<JPH::Ref<JPH::TwoBodyConstraint>> JoltItem::get_constraints() { return {}; }

    JoltPhysicEngine &JoltItem::get_engine() const { return engine; }

}// namespace arenai::model
