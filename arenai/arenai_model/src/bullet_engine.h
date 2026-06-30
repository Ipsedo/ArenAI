//
// Created by claude on 30/06/2026.
//

#ifndef ARENAI_BULLET_ENGINE_H
#define ARENAI_BULLET_ENGINE_H

#include <mutex>
#include <shared_mutex>

#include <btBulletDynamicsCommon.h>

#include <arenai_model/engine.h>

#include "bullet_item.h"

class BulletPhysicEngine final : public AbstractPhysicEngine {
public:
    explicit BulletPhysicEngine(float wanted_frequency);

    void add_item(const std::shared_ptr<Item> &item) override;
    void add_item_producer(const std::shared_ptr<ItemProducer> &item_producer) override;
    void remove_item_constraints_from_world(const std::shared_ptr<Item> &item) override;

    void step(float delta) override;

    std::vector<std::shared_ptr<Item>> get_items() override;

    void remove_bodies_and_constraints() override;

    ~BulletPhysicEngine() override;

private:
    std::shared_mutex items_mutex;

    float wanted_frequency;

    btDefaultCollisionConfiguration *m_collision_configuration;
    btCollisionDispatcher *m_dispatcher;
    btBroadphaseInterface *m_broad_phase;
    btSequentialImpulseConstraintSolver *m_constraint_solver;
    btDiscreteDynamicsWorld *m_world;

    std::vector<std::shared_ptr<BulletItem>> items;
    std::vector<std::shared_ptr<ItemProducer>> item_producers;

    void add_bullet_item(const std::shared_ptr<BulletItem> &item);
    void remove_dead_items();
};

#endif// ARENAI_BULLET_ENGINE_H
