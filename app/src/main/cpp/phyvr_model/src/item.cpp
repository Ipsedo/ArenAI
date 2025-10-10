//
// Created by samuel on 19/03/2023.
//

#include <glm/gtc/type_ptr.hpp>

#include <phyvr_model/item.h>

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

LifeItem::LifeItem(int health_points) : health_points(health_points) {}

bool LifeItem::is_dead() const { return health_points <= 0; }

void LifeItem::receive_damages(int damages) { health_points -= damages; }
