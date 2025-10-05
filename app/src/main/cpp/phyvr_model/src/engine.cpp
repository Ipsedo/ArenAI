//
// Created by samuel on 19/03/2023.
//

#include <algorithm>

#include <phyvr_model/engine.h>

static btITaskScheduler *get_or_create_task_scheduler() {
    auto scheduler = btGetTaskScheduler();
    if (!scheduler) {
        scheduler = btCreateDefaultTaskScheduler();
        btSetTaskScheduler(scheduler);
    }
    scheduler->activate();
    return scheduler;
}

InitBtThread::InitBtThread(const int num_threads) : task_scheduler(get_or_create_task_scheduler()), cci() {
  task_scheduler->setNumThreads(num_threads);

  cci.m_defaultMaxPersistentManifoldPoolSize = 8192;
  cci.m_defaultMaxCollisionAlgorithmPoolSize = 8192;
}

btDefaultCollisionConstructionInfo InitBtThread::get_cci() const { return cci; }

InitBtThread::~InitBtThread() {
    task_scheduler->deactivate();
}

PhysicEngine::PhysicEngine(int threads_num)
    : init_thread(std::make_unique<InitBtThread>(threads_num)),
      m_collision_configuration(new btDefaultCollisionConfiguration(init_thread->get_cci())),
      m_dispatcher(new btCollisionDispatcherMt(m_collision_configuration, 40)),
      m_broad_phase(new btDbvtBroadphase()),
      m_pool_solver(new btConstraintSolverPoolMt(threads_num)),
      m_constraint_solver(new btSequentialImpulseConstraintSolverMt()),
      m_world(new btDiscreteDynamicsWorldMt(
        m_dispatcher, m_broad_phase, m_pool_solver, m_constraint_solver,
        m_collision_configuration)),
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

  m_world->stepSimulation(delta, 1, 1.f / 60.f);
}

std::vector<std::shared_ptr<Item>> PhysicEngine::get_items() { return items; }

void PhysicEngine::remove_bodies_and_constraints() {
    for (int i = m_world->getNumCollisionObjects() - 1; i >= 0; i--) {
        btCollisionObject *obj = m_world->getCollisionObjectArray()[i];
        m_world->removeCollisionObject(obj);
        const auto body = btRigidBody::upcast(obj);

        while (body->getNumConstraintRefs()) {
            btTypedConstraint *constraint = body->getConstraintRef(0);
            m_world->removeConstraint(constraint);
            delete constraint;
        }

        m_world->removeRigidBody(body);

        auto motion_state = body->getMotionState();
        delete motion_state;

        delete body;
    }
}

PhysicEngine::~PhysicEngine() {
  item_producers.clear();
  items.clear();
  remove_bodies_and_constraints();

  delete m_world;
  delete m_pool_solver;
  delete m_broad_phase;
  delete m_dispatcher;
  delete m_collision_configuration;
  delete m_constraint_solver;
}
