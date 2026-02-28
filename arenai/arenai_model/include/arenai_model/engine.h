//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_ENGINE_H
#define ARENAI_ENGINE_H

#include <mutex>
#include <tuple>
#include <vector>

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>

#include "./item.h"

class PhysicEngine {
public:
    explicit PhysicEngine(float wanted_frequency);

    void add_item(const std::shared_ptr<Item> &item);
    void add_item_producer(const std::shared_ptr<ItemProducer> &item_producer);
    void remove_item_constraints_from_world(const std::shared_ptr<Item> &item);

    void step(float delta);

    std::vector<std::shared_ptr<Item>> get_items();

    void remove_bodies_and_constraints();

    ~PhysicEngine();

private:
    std::mutex items_mutex;

    float wanted_frequency;

    btDefaultCollisionConfiguration *m_collision_configuration;
    btCollisionDispatcher *m_dispatcher;
    btBroadphaseInterface *m_broad_phase;
    btSequentialImpulseConstraintSolver *m_constraint_solver;
    btDiscreteDynamicsWorld *m_world;

    std::vector<std::shared_ptr<Item>> items;
    std::vector<std::shared_ptr<ItemProducer>> item_producers;

    void remove_dead_items();
    void add_item_no_lock(const std::shared_ptr<Item> &item);
};

#endif// ARENAI_ENGINE_H
