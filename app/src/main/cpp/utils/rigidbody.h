//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_RIGODBODY_H
#define PHYVR_RIGODBODY_H

#include <tuple>
#include <btBulletDynamicsCommon.h>
#include <string>

using namespace std;

tuple<btRigidBody *, btDefaultMotionState *>
localCreateRigidBody(btScalar mass, const btTransform &startTransform, btCollisionShape *shape);
btConvexHullShape *parseObj(string objFileText);

#endif //PHYVR_RIGODBODY_H
