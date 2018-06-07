//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_RIGODBODY_H
#define PHYVR_RIGODBODY_H

#include <tuple>
#include <btBulletDynamicsCommon.h>
#include <string>
#include "../entity/base.h"

using namespace std;

tuple<btRigidBody *, btDefaultMotionState *>
localCreateRigidBody(btScalar mass, const btTransform &startTransform, btCollisionShape *shape, Base* b);
btConvexHullShape *parseObj(string objFileText);

class btRigidBodyWithBase : public btRigidBody {
public:
	btRigidBodyWithBase(btScalar mass, btMotionState *motionState, btCollisionShape *collisionShape,
						const btVector3 &localInertia, Base* b);

	btRigidBodyWithBase(const btRigidBodyConstructionInfo &constructionInfo, Base* affectedBase);
	Base* base;
};

#endif //PHYVR_RIGODBODY_H
