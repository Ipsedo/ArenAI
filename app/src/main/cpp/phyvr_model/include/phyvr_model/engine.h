//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_ENGINE_H
#define PHYVR_ENGINE_H

#include <tuple>
#include <vector>

#include <btBulletDynamicsCommon.h>
#include <BulletCollision/CollisionDispatch/btCollisionDispatcherMt.h>
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h>
#include <BulletDynamics/Dynamics/btDiscreteDynamicsWorldMt.h>
#include <glm/glm.hpp>

#include "./item.h"

class InitBtThread {
public:
  explicit InitBtThread(int num_threads);
  btDefaultCollisionConstructionInfo get_cci() const;

    virtual ~InitBtThread();

private:
    btITaskScheduler *task_scheduler;
  btDefaultCollisionConstructionInfo cci;
};

class PhysicEngine {
public:
  explicit PhysicEngine(int threads_num);

  void add_item(const std::shared_ptr<Item> &item);
  void add_item_producer(const std::shared_ptr<ItemProducer> &item_producer);

  void step(float delta);

  std::vector<std::shared_ptr<Item>> get_items();

  void remove_bodies_and_constraints();

  ~PhysicEngine();

private:
  std::unique_ptr<InitBtThread> init_thread;
  btDefaultCollisionConfiguration *m_collision_configuration;
  btCollisionDispatcherMt *m_dispatcher;
  btBroadphaseInterface *m_broad_phase;
  btConstraintSolverPoolMt *m_pool_solver;
  btSequentialImpulseConstraintSolverMt *m_constraint_solver;
  btDiscreteDynamicsWorldMt *m_world;

  std::vector<std::shared_ptr<Item>> items;
  std::vector<std::shared_ptr<ItemProducer>> item_producers;
};

#endif// PHYVR_ENGINE_H
