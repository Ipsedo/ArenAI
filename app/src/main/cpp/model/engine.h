//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_ENGINE_H
#define PHYVR_ENGINE_H

#include <btBulletDynamicsCommon.h>
#include <glm/glm.hpp>
#include <tuple>
#include <vector>

#include "./item.h"

class PhysicEngine {
public:
  PhysicEngine();

  void add_item(const std::shared_ptr<Item> &item);

  void step(float delta);

  ~PhysicEngine();

private:
  btDefaultCollisionConfiguration *m_collision_configuration;
  btCollisionDispatcher *m_dispatcher;
  btBroadphaseInterface *m_broad_phase;
  btSequentialImpulseConstraintSolver *m_constraint_solver;
  btDiscreteDynamicsWorld *m_world;
};

#endif // PHYVR_ENGINE_H
