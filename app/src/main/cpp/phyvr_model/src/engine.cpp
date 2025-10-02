//
// Created by samuel on 19/03/2023.
//

#include <phyvr_model/engine.h>

#include <algorithm>

InitBtThread::InitBtThread(const int num_threads) {
  btSetTaskScheduler(btCreateDefaultTaskScheduler());
  btGetTaskScheduler()->setNumThreads(num_threads);
  cci.m_defaultMaxPersistentManifoldPoolSize = 8192;
  cci.m_defaultMaxCollisionAlgorithmPoolSize = 8192;
}

btDefaultCollisionConstructionInfo InitBtThread::get_cci() const { return cci; }

PhysicEngine::PhysicEngine(int threads_num)
    : threads_num(threads_num), init_thread(threads_num),
      m_collision_configuration(
          new btDefaultCollisionConfiguration(init_thread.get_cci())),
      m_dispatcher(new btCollisionDispatcherMt(m_collision_configuration, 40)),
      m_broad_phase(new btDbvtBroadphase()),
      m_pool_solver(new btConstraintSolverPoolMt(threads_num)),
      m_constraint_solver(new btSequentialImpulseConstraintSolverMt()),
      m_world(new btDiscreteDynamicsWorldMt(m_dispatcher, m_broad_phase,
                                            m_pool_solver, m_constraint_solver,
                                            m_collision_configuration)),
      item_producers(), items() {

  m_world->setGravity(btVector3(0, -9.8f, 0));
}

void PhysicEngine::add_item(const std::shared_ptr<Item> &item) {
  items.push_back(item);

  m_world->addRigidBody(item->get_body());

  for (auto &constraint : item->get_constraints())
    m_world->addConstraint(constraint, true);
}

void PhysicEngine::add_item_producer(
    const std::shared_ptr<ItemProducer> &item_producer) {
  item_producers.push_back(item_producer);
}

void PhysicEngine::step(float delta) {
  for (const auto &item_producer : item_producers)
    for (const auto &item : item_producer->get_produced_items())
      add_item(item);

  m_world->stepSimulation(delta, 1, 1.f / 60.f);
}

std::vector<std::shared_ptr<Item>> PhysicEngine::get_items() { return items; }

void PhysicEngine::remove_bodies_and_constraints() {
  for (int i = m_world->getNumConstraints() - 1; i >= 0; i--) {
    btTypedConstraint *constraint = m_world->getConstraint(i);
    m_world->removeConstraint(constraint);
    delete constraint;
  }

  for (int i = m_world->getNumCollisionObjects() - 1; i >= 0; i--) {
    btCollisionObject *obj = m_world->getCollisionObjectArray()[i];
    m_world->removeCollisionObject(obj);
    delete obj;
  }
}

PhysicEngine::~PhysicEngine() {
  remove_bodies_and_constraints();

  delete m_world;
  delete m_broad_phase;
  delete m_dispatcher;
  delete m_collision_configuration;
  delete m_constraint_solver;
}
