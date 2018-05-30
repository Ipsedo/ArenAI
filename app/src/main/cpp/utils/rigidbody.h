//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_RIGODBODY_H
#define PHYVR_RIGODBODY_H

#include <tuple>
#include <btBulletDynamicsCommon.h>

std::tuple<btRigidBody *, btDefaultMotionState *>
localCreateRigidBody(btScalar mass, const btTransform &startTransform, btCollisionShape *shape);

#endif //PHYVR_RIGODBODY_H
