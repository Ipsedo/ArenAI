//
// Created by samuel on 19/03/2023.
//

#include "engine.h"

#include <algorithm>

Engine::Engine() :
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

void Engine::add_item(const std::shared_ptr<Item> &item) {
    items.push_back(item);
    m_world->addRigidBody(item->get_body());
}

void Engine::step(float delta) {
    m_world->stepSimulation(delta);
}
