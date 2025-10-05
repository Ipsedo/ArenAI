//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_ITEM_H
#define PHYVR_ITEM_H

#include <memory>
#include <string>

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>

#include "./shapes.h"

class Item {
public:
    Item(std::string name);

    virtual std::shared_ptr<Shape> get_shape() = 0;

    std::string get_name();

    glm::mat4 get_model_matrix();

    virtual btRigidBody *get_body() = 0;
    virtual std::vector<btTypedConstraint *> get_constraints();

    virtual void on_contact(Item *other);
    virtual bool need_destroy();

private:
    std::string name;

protected:
    virtual glm::vec3 _get_scale() = 0;
};

class ItemProducer {
public:
    virtual std::vector<std::shared_ptr<Item>> get_produced_items() = 0;
};

#endif// PHYVR_ITEM_H
