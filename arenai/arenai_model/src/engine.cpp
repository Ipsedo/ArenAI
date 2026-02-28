//
// Created by samuel on 19/03/2023.
//

#include <algorithm>

#include <arenai_model/engine.h>

#include "tank/shell.h"

PhysicEngine::PhysicEngine(const float wanted_frequency)
    : wanted_frequency(wanted_frequency),
      m_collision_configuration(new btDefaultCollisionConfiguration()),
      m_dispatcher(new btCollisionDispatcher(m_collision_configuration)),
      m_broad_phase(new btDbvtBroadphase()),
      m_constraint_solver(new btSequentialImpulseConstraintSolver()),
      m_world(new btDiscreteDynamicsWorld(
          m_dispatcher, m_broad_phase, m_constraint_solver, m_collision_configuration)) {

    m_world->setGravity(btVector3(0, -9.8f, 0));
}

void PhysicEngine::add_item(const std::shared_ptr<Item> &item) {
    std::scoped_lock lock(items_mutex);
    add_item_no_lock(item);
}

void PhysicEngine::add_item_no_lock(const std::shared_ptr<Item> &item) {
    items.push_back(item);

    m_world->addRigidBody(item->get_body());

    for (const auto &constraint: item->get_constraints()) m_world->addConstraint(constraint, true);
}

void PhysicEngine::add_item_producer(const std::shared_ptr<ItemProducer> &item_producer) {
    std::scoped_lock lock(items_mutex);
    item_producers.push_back(item_producer);
}

void PhysicEngine::remove_item_constraints_from_world(const std::shared_ptr<Item> &item) {
    std::scoped_lock lock(items_mutex);
    for (auto *constraint: item->get_constraints()) m_world->removeConstraint(constraint);
}

void PhysicEngine::step(const float delta) {
    {
        std::scoped_lock lock(items_mutex);
        for (const auto &item_producer: item_producers)
            for (const auto &item: item_producer->get_produced_items()) add_item_no_lock(item);
    }

    m_world->stepSimulation(delta, 1, wanted_frequency);

    for (int i = 0; i < m_dispatcher->getNumManifolds(); i++) {
        btPersistentManifold *manifold = m_dispatcher->getManifoldByIndexInternal(i);
        const auto a = btRigidBody::upcast(manifold->getBody0());
        const auto b = btRigidBody::upcast(manifold->getBody1());

        const int num_contacts = manifold->getNumContacts();
        for (int j = 0; j < num_contacts; j++) {
            if (const btManifoldPoint &pt = manifold->getContactPoint(j); pt.getDistance() < 0.0f) {
                const auto item_a = static_cast<Item *>(a->getUserPointer());
                const auto item_b = static_cast<Item *>(b->getUserPointer());

                item_a->on_contact(item_b);
                item_b->on_contact(item_a);
            }
        }
    }

    remove_dead_items();
}

std::vector<std::shared_ptr<Item>> PhysicEngine::get_items() {
    std::scoped_lock lock(items_mutex);
    return items;
}

void PhysicEngine::remove_dead_items() {
    std::scoped_lock lock(items_mutex);

    for (int i = static_cast<int>(items.size()) - 1; i >= 0; i--) {
        const auto item = items[i];

        item->tick();

        if (item->need_destroy()) {
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

void PhysicEngine::remove_bodies_and_constraints() {
    std::scoped_lock lock(items_mutex);
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

    item_producers.clear();
    items.clear();
}

PhysicEngine::~PhysicEngine() {
    remove_bodies_and_constraints();

    delete m_world;
    delete m_constraint_solver;
    delete m_dispatcher;
    delete m_broad_phase;
    delete m_collision_configuration;
}
