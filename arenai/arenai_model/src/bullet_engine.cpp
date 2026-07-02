//
// Created by claude on 30/06/2026.
//

#include "./bullet_engine.h"

#include "./items/bullet_item_factory.h"

using namespace arenai;
using namespace arenai::model;

namespace arenai::model {

    BulletPhysicEngine::BulletPhysicEngine(const float wanted_frequency)
        : wanted_frequency(wanted_frequency) {
        m_collision_configuration = new btDefaultCollisionConfiguration();
        m_dispatcher = new btCollisionDispatcher(m_collision_configuration);
        m_broad_phase = new btDbvtBroadphase();
        m_constraint_solver = new btSequentialImpulseConstraintSolver();
        m_world = new btDiscreteDynamicsWorld(
            m_dispatcher, m_broad_phase, m_constraint_solver, m_collision_configuration);

        m_world->setGravity(btVector3(0, -9.8f, 0));

        item_factory = std::make_shared<BulletItemFactory>(*this);
    }

    void BulletPhysicEngine::add_bullet_item(const std::shared_ptr<BulletItem> &item) {
        std::unique_lock lock(items_mutex);
        items.push_back(item);

        m_world->addRigidBody(item->get_body());

        for (const auto &constraint: item->get_constraints())
            m_world->addConstraint(constraint, true);
    }

    void BulletPhysicEngine::add_bullet_item_producer(
        std::function<std::vector<std::shared_ptr<BulletItem>>()> producer) {
        std::unique_lock lock(items_mutex);
        bullet_item_producers.push_back(std::move(producer));
    }

    void
    BulletPhysicEngine::remove_bullet_item_constraints(const std::shared_ptr<BulletItem> &item) {
        std::unique_lock lock(items_mutex);
        for (auto *constraint: item->get_constraints()) m_world->removeConstraint(constraint);
    }

    void BulletPhysicEngine::remove_dead_items() {
        std::unique_lock lock(items_mutex);

        for (const auto &item: items) item->tick();

        for (int i = static_cast<int>(items.size()) - 1; i >= 0; i--) {
            if (const auto item = items[i]; item->need_destroy()) {
                auto *body = item->get_body();

                m_world->removeRigidBody(body);

                for (int j = body->getNumConstraintRefs() - 1; j >= 0; j--) {
                    auto *constraint = body->getConstraintRef(j);
                    m_world->removeConstraint(constraint);
                    delete constraint;
                }

                delete body->getMotionState();

                delete body;

                items.erase(items.begin() + i);
            }
        }
    }

    void BulletPhysicEngine::step(const float delta) {
        {
            std::unique_lock lock(items_mutex);
            for (const auto &producer: bullet_item_producers)
                for (const auto &item: producer()) {
                    items.push_back(item);
                    m_world->addRigidBody(item->get_body());
                    for (const auto &constraint: item->get_constraints())
                        m_world->addConstraint(constraint, true);
                }
        }

        m_world->stepSimulation(delta, 1, wanted_frequency);

        for (int i = 0; i < m_dispatcher->getNumManifolds(); i++) {
            btPersistentManifold *manifold = m_dispatcher->getManifoldByIndexInternal(i);
            const auto a = btRigidBody::upcast(manifold->getBody0());
            const auto b = btRigidBody::upcast(manifold->getBody1());

            const int num_contacts = manifold->getNumContacts();
            for (int j = 0; j < num_contacts; j++) {
                if (const btManifoldPoint &pt = manifold->getContactPoint(j);
                    pt.getDistance() < 0.0f) {
                    const auto item_a = static_cast<Item *>(a->getUserPointer());
                    const auto item_b = static_cast<Item *>(b->getUserPointer());

                    item_a->on_contact(item_b);
                    item_b->on_contact(item_a);
                }
            }
        }

        remove_dead_items();
    }

    std::vector<std::shared_ptr<Item>> BulletPhysicEngine::get_items() {
        std::shared_lock lock(items_mutex);
        return {items.begin(), items.end()};
    }

    std::shared_ptr<ItemFactory> BulletPhysicEngine::get_item_factory() { return item_factory; }

    void BulletPhysicEngine::remove_bodies_and_constraints() {
        std::unique_lock lock(items_mutex);
        m_world->clearForces();

        for (int i = m_world->getNumConstraints() - 1; i >= 0; --i) {
            btTypedConstraint *constraint = m_world->getConstraint(i);
            m_world->removeConstraint(constraint);
        }

        for (int i = m_world->getNumCollisionObjects() - 1; i >= 0; --i) {
            btRigidBody *body = btRigidBody::upcast(m_world->getCollisionObjectArray()[i]);
            m_world->removeRigidBody(body);
        }

        for (const auto &item: items) {
            for (const auto *constraint: item->get_constraints()) delete constraint;

            auto *body = item->get_body();
            delete body->getMotionState();
            delete body;
        }

        bullet_item_producers.clear();
        items.clear();
    }

    BulletPhysicEngine::~BulletPhysicEngine() {
        remove_bodies_and_constraints();

        delete m_world;
        delete m_constraint_solver;
        delete m_dispatcher;
        delete m_broad_phase;
        delete m_collision_configuration;
    }

}// namespace arenai::model
