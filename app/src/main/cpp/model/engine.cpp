//
// Created by samuel on 19/03/2023.
//

#include "./engine.h"

#include <algorithm>

PhysicEngine::PhysicEngine() :
        m_collision_configuration(new btDefaultCollisionConfiguration()),
        m_dispatcher(new btCollisionDispatcher(m_collision_configuration)),
        m_broad_phase(new btDbvtBroadphase()),
        m_constraint_solver(new btSequentialImpulseConstraintSolver()),
        m_world(new btDiscreteDynamicsWorld(
                m_dispatcher,
                m_broad_phase,
                m_constraint_solver,
                m_collision_configuration)) {

    m_world->setGravity(btVector3(0, -9.8f, 0));

}

void PhysicEngine::add_item(const std::shared_ptr<Item> &item) {
    m_world->addRigidBody(item->get_body());
}

void PhysicEngine::step(float delta) {
    m_world->stepSimulation(delta);
}

PhysicEngine::~PhysicEngine() {
    for (int i = m_world->getNumCollisionObjects() - 1; i >= 0; i--) {
        btCollisionObject *obj = m_world->getCollisionObjectArray()[i];
        m_world->removeCollisionObject(obj);
        delete obj;
    }

    delete m_world;
    delete m_broad_phase;
    delete m_dispatcher;
    delete m_collision_configuration;
    delete m_constraint_solver;
}
