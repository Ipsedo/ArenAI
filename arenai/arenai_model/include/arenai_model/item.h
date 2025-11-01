//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_ITEM_H
#define ARENAI_ITEM_H

#include <memory>
#include <string>

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>

#include "./shapes.h"

class Item {
public:
    virtual ~Item() = default;

    explicit Item(std::string name);

    virtual std::shared_ptr<Shape> get_shape() = 0;

    std::string get_name();

    glm::mat4 get_model_matrix();

    virtual void tick();

    virtual btRigidBody *get_body() = 0;
    virtual std::vector<btTypedConstraint *> get_constraints();

    virtual void on_contact(Item *other);
    virtual bool need_destroy();
    virtual void destroy();

private:
    std::string name;
    bool will_destroy;

protected:
    virtual glm::vec3 _get_scale() = 0;
};

class ItemProducer {
public:
    virtual ~ItemProducer() = default;

    virtual std::vector<std::shared_ptr<Item>> get_produced_items() = 0;
};

class LifeItem {
public:
    virtual ~LifeItem() = default;

    explicit LifeItem(float health_points);

    bool is_dead() const;
    bool is_already_dead();

    float receive_damages(float damages);

private:
    float health_points;
    bool already_dead;
};

#endif// ARENAI_ITEM_H
