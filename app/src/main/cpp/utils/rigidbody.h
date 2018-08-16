//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_RIGODBODY_H
#define PHYVR_RIGODBODY_H

#include <tuple>
#include <glm/glm.hpp>
#include <btBulletDynamicsCommon.h>
#include <string>
#include <glm/gtc/quaternion.hpp>
#include <android/asset_manager.h>

using namespace std;

btRigidBody::btRigidBodyConstructionInfo
localCreateInfo(btScalar mass, const btTransform &startTransform, btCollisionShape *shape);

btConvexHullShape *parseObj(string objFileText);

template<typename F>
btRigidBody::btRigidBodyConstructionInfo
makeCInfo(F fun, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass) {
	btCollisionShape *shape = fun();

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotMat);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	return localCreateInfo(mass, myTransform, shape);
};

template<class C, typename FunShape, typename FunBase>
C *makeBase(AAssetManager *mgr, FunShape f1, FunBase f2, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass) {

	btCollisionShape *shape = f1(scale);

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotMat);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	btRigidBody::btRigidBodyConstructionInfo cinfo = localCreateInfo(mass, myTransform, shape);

	return f2(mgr, cinfo, scale);
};

/*btRigidBody::btRigidBodyConstructionInfo
createRBInfo(btCollisionShape shape, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat)*/

#endif //PHYVR_RIGODBODY_H
