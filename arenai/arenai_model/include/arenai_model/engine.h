//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_ENGINE_H
#define ARENAI_ENGINE_H

#include <memory>
#include <vector>

#include <arenai_utils/file_reader.h>

#include "./item.h"

class ItemFactory;
class TankFactory;

class AbstractPhysicEngine {
public:
    virtual ~AbstractPhysicEngine() = default;

    virtual void step(float delta) = 0;

    virtual std::vector<std::shared_ptr<Item>> get_items() = 0;

    virtual void remove_bodies_and_constraints() = 0;

    virtual std::shared_ptr<ItemFactory> get_item_factory() = 0;
};

std::unique_ptr<AbstractPhysicEngine> make_physic_engine(float wanted_frequency);

std::shared_ptr<TankFactory> make_tank_factory(
    AbstractPhysicEngine &engine, const std::shared_ptr<AbstractFileReader> &file_reader,
    float wanted_frame_frequency);

#endif// ARENAI_ENGINE_H
