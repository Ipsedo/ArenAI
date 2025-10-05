//
// Created by samuel on 23/03/20.
//

#ifndef PHYVR_SOLID_CURVE_H
#define PHYVR_SOLID_CURVE_H

#include "../entity/tank/canon.h"
#include "../graphics/misc.h"
#include "btBulletDynamicsCommon.h"

class Curve : public Drawable {
private:
  Canon *canon;

  // GPU program handle
  GLuint mProgram;
  GLuint vertexShader;
  GLuint fragmentShader;

  GLuint mPositionHandle;
  GLuint mVPMatrixHandle;
  GLuint mColorHandle;

  const int nb_pts = 120;

  btBroadphaseInterface *broadPhase;
  btCollisionDispatcher *dispatcher;
  btDefaultCollisionConfiguration *collisionConfiguration;
  btSequentialImpulseConstraintSolver *constraintSolver;
  btDiscreteDynamicsWorld *world;

public:
  Curve(Canon *canon);

  void draw(draw_infos infos) override;
};

std::vector<btTransform> get_trajectory(btRigidBody *body);

#endif// PHYVR_SOLID_CURVE_H
