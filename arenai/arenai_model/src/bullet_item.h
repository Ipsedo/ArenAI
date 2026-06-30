//
// Created by claude on 30/06/2026.
//

#ifndef ARENAI_BULLET_ITEM_H
#define ARENAI_BULLET_ITEM_H

#include <btBulletDynamicsCommon.h>

#include <arenai_model/item.h>

class BulletItem : public Item {
public:
    using Item::Item;

    glm::mat4 get_model_matrix() override;

    glm::vec3 get_linear_velocity() override;
    glm::vec3 get_angular_velocity() override;

    virtual btRigidBody *get_body() = 0;
    virtual std::vector<btTypedConstraint *> get_constraints();
};

#endif// ARENAI_BULLET_ITEM_H
