//
// Created by samuel on 29/05/18.
//

#include "rigidbody.h"


std::tuple<btRigidBody *, btDefaultMotionState *>
localCreateRigidBody(btScalar mass, const btTransform &startTransform, btCollisionShape *shape) {
	btAssert((!shape || shape->getShapeType() != INVALID_SHAPE_PROXYTYPE));

	//rigidbody is dynamic if and only if mass is non zero, otherwise static
	bool isDynamic = (mass != 0.f);

	btVector3 localInertia(0, 0, 0);
	if (isDynamic)
		shape->calculateLocalInertia(mass, localInertia);

	btDefaultMotionState *myMotionState = new btDefaultMotionState(startTransform);

	btRigidBody::btRigidBodyConstructionInfo cInfo(mass, myMotionState, shape, localInertia);

	return std::tuple<btRigidBody *, btDefaultMotionState *>(new btRigidBody(cInfo), myMotionState);
}