//
// Created by claude on 30/06/2026.
//

#ifndef ARENAI_BULLET_ENGINE_H
#define ARENAI_BULLET_ENGINE_H

#include <functional>
#include <mutex>
#include <shared_mutex>

#include <btBulletDynamicsCommon.h>

#include <arenai_model/engine.h>
#include <arenai_model/item_factory.h>

#include "./bullet_item.h"

namespace arenai::model {

    class BulletPhysicEngine final : public AbstractPhysicEngine {
    public:
        explicit BulletPhysicEngine(float wanted_frequency);

        void step(float delta) override;

        std::vector<std::shared_ptr<Item>> get_items() override;

        void remove_bodies_and_constraints() override;

        std::shared_ptr<ItemFactory> get_item_factory() override;

        ~BulletPhysicEngine() override;

        void add_bullet_item(const std::shared_ptr<BulletItem> &item);
        void add_bullet_item_producer(
            std::function<std::vector<std::shared_ptr<BulletItem>>()> producer);
        void remove_bullet_item_constraints(const std::shared_ptr<BulletItem> &item);

    private:
        std::shared_mutex items_mutex;

        float wanted_frequency;

        btDefaultCollisionConfiguration *m_collision_configuration;
        btCollisionDispatcher *m_dispatcher;
        btBroadphaseInterface *m_broad_phase;
        btSequentialImpulseConstraintSolver *m_constraint_solver;
        btDiscreteDynamicsWorld *m_world;

        std::vector<std::shared_ptr<BulletItem>> items;
        std::vector<std::function<std::vector<std::shared_ptr<BulletItem>>()>>
            bullet_item_producers;

        std::shared_ptr<ItemFactory> item_factory;

        void remove_dead_items();
    };

}// namespace arenai::model

#endif// ARENAI_BULLET_ENGINE_H
