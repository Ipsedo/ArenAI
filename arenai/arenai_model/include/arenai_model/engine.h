//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_ENGINE_H
#define ARENAI_ENGINE_H

#include <memory>
#include <vector>

#include "./item.h"

class AbstractPhysicEngine {
public:
    virtual ~AbstractPhysicEngine() = default;

    virtual void add_item(const std::shared_ptr<Item> &item) = 0;
    virtual void add_item_producer(const std::shared_ptr<ItemProducer> &item_producer) = 0;
    virtual void remove_item_constraints_from_world(const std::shared_ptr<Item> &item) = 0;

    virtual void step(float delta) = 0;

    virtual std::vector<std::shared_ptr<Item>> get_items() = 0;

    virtual void remove_bodies_and_constraints() = 0;
};

std::unique_ptr<AbstractPhysicEngine> make_physic_engine(float wanted_frequency);

#endif// ARENAI_ENGINE_H
