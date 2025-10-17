//
// Created by samuel on 19/03/2023.
//

#include <glm/gtc/type_ptr.hpp>

#include <arenai_model/item.h>

/*
 * Base Item
 */

Item::Item(std::string name) : name(std::move(name)), will_destroy(false) {}

std::string Item::get_name() { return name; }

glm::mat4 Item::get_model_matrix() {
    btScalar tmp[16];
    btTransform tr;

    get_body()->getMotionState()->getWorldTransform(tr);

    tr.getOpenGLMatrix(tmp);

    return glm::make_mat4(tmp) * glm::scale(glm::mat4(1.f), _get_scale());
}

std::vector<btTypedConstraint *> Item::get_constraints() { return {}; }

bool Item::need_destroy() { return will_destroy; }

void Item::destroy() { will_destroy = true; }

void Item::on_contact(Item *other) {}

/*
 * Life Item
 */

LifeItem::LifeItem(const float health_points) : health_points(health_points), already_dead(false) {}

bool LifeItem::is_dead() const { return health_points <= 0.f; }

bool LifeItem::is_already_dead() {
    if (is_dead() && !already_dead) {
        already_dead = true;
        return false;
    }
    return already_dead;
}

float LifeItem::receive_damages(const float damages) {
    const float new_health_point = std::max(health_points - damages, 0.f);
    const float received_damages = health_points - new_health_point;

    health_points = new_health_point;

    return received_damages;
}
