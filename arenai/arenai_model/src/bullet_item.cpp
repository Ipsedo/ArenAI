//
// Created by claude on 30/06/2026.
//

#include "./bullet_item.h"

#include <glm/gtc/type_ptr.hpp>

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    glm::mat4 BulletItem::get_model_matrix() {
        btScalar tmp[16];
        btTransform tr;

        get_body()->getMotionState()->getWorldTransform(tr);

        tr.getOpenGLMatrix(tmp);

        return glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), _get_scale());
    }

    glm::vec3 BulletItem::get_linear_velocity() {
        const auto vel = get_body()->getLinearVelocity();
        return {vel.x(), vel.y(), vel.z()};
    }

    glm::vec3 BulletItem::get_angular_velocity() {
        const auto vel = get_body()->getAngularVelocity();
        return {vel.x(), vel.y(), vel.z()};
    }

    std::vector<btTypedConstraint *> BulletItem::get_constraints() { return {}; }

}// namespace arenai::model
