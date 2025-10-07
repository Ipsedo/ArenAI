//
// Created by samuel on 19/03/2023.
//

#include <algorithm>

#include <phyvr_model/engine.h>

PhysicEngine::PhysicEngine(float wanted_frequency)
    : wanted_frequency(wanted_frequency),
      m_collision_configuration(new btDefaultCollisionConfiguration()),
      m_dispatcher(new btCollisionDispatcher(m_collision_configuration)),
      m_broad_phase(new btDbvtBroadphase()),
      m_constraint_solver(new btSequentialImpulseConstraintSolver()),
      m_world(new btDiscreteDynamicsWorld(
          m_dispatcher, m_broad_phase, m_constraint_solver, m_collision_configuration)),
      item_producers(), items() {

    m_world->setGravity(btVector3(0, -9.8f, 0));
}

void PhysicEngine::add_item(const std::shared_ptr<Item> &item) {
    items.push_back(item);

    m_world->addRigidBody(item->get_body());

    for (auto &constraint: item->get_constraints()) m_world->addConstraint(constraint, true);
}

void PhysicEngine::add_item_producer(const std::shared_ptr<ItemProducer> &item_producer) {
    item_producers.push_back(item_producer);
}

void PhysicEngine::step(float delta) {
    for (const auto &item_producer: item_producers)
        for (const auto &item: item_producer->get_produced_items()) add_item(item);

    m_world->stepSimulation(delta, 1, wanted_frequency);

    for (int i = 0; i < m_dispatcher->getNumManifolds(); i++) {
        btPersistentManifold *manifold = m_dispatcher->getManifoldByIndexInternal(i);
        auto a = btRigidBody::upcast(manifold->getBody0());
        auto b = btRigidBody::upcast(manifold->getBody1());

        int num_contacts = manifold->getNumContacts();
        for (int j = 0; j < num_contacts; j++) {
            const btManifoldPoint &pt = manifold->getContactPoint(j);
            if (pt.getDistance() < 0.0f) {
                auto item_a = (Item *) a->getUserPointer();
                auto item_b = (Item *) b->getUserPointer();

                item_a->on_contact(item_b);
                item_b->on_contact(item_a);
            }
        }
    }

    for (int i = static_cast<int>(items.size()) - 1; i >= 0; i--) {
        auto item = items[i];
        if (item->need_destroy()) {
            auto body = item->get_body();
            m_world->removeCollisionObject(body);
            m_world->removeRigidBody(body);

            for (int j = body->getNumConstraintRefs() - 1; j >= 0; j--)
                m_world->removeConstraint(body->getConstraintRef(j));

            items.erase(items.begin() + i);
        }
    }
}

std::vector<std::shared_ptr<Item>> PhysicEngine::get_items() { return items; }

void PhysicEngine::remove_bodies_and_constraints() {
    item_producers.clear();
    items.clear();

    m_world->clearForces();

    for (int i = m_world->getNumConstraints() - 1; i >= 0; --i) {
        btTypedConstraint *constraint = m_world->getConstraint(i);
        m_world->removeConstraint(constraint);
        delete constraint;
    }

    for (int i = m_world->getNumCollisionObjects() - 1; i >= 0; --i) {
        btCollisionObject *obj = m_world->getCollisionObjectArray()[i];
        btRigidBody *body = btRigidBody::upcast(obj);
        if (body && body->getMotionState()) { delete body->getMotionState(); }
        m_world->removeCollisionObject(obj);
        delete obj;
    }
}

PhysicEngine::~PhysicEngine() {
    remove_bodies_and_constraints();

    delete m_world;
    delete m_constraint_solver;
    delete m_dispatcher;
    delete m_broad_phase;
    delete m_collision_configuration;
}
