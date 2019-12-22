//
// Created by samuel on 21/12/19.
//

#include "poly.h"

#include "../utils/assets.h"
#include "../utils/rigidbody.h"

#include <glm/gtc/quaternion.hpp>


btRigidBody::btRigidBodyConstructionInfo
Poly::makeCInfo(std::function<btCollisionShape *(glm::vec3)> makeShapeFun,
		glm::vec3 pos, glm::mat4 rotMat, glm::vec3 scale, float mass) {
	btCollisionShape *shape = makeShapeFun(scale);

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotMat);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	return localCreateInfo(mass, myTransform, shape);
}

Poly::Poly(std::function<btCollisionShape *(glm::vec3)> makeShapeFun,
		GLDrawable *modelVBO, const glm::vec3 pos, const glm::vec3 &scale,
		glm::mat4 rotMat, float mass, bool hasOwnModel)
		: Base(makeCInfo(std::move(makeShapeFun), pos, rotMat, scale, mass),
				modelVBO, scale, hasOwnModel) {}

DiffuseModel *Poly::makeModel(AAssetManager *mgr, string objFileName) {
	string objTxt = getFileText(mgr, std::move(objFileName));
	// TODO #include <random>
	return new ModelVBO(objTxt,
						(float) rand() / RAND_MAX,
						(float) rand() / RAND_MAX,
						(float) rand() / RAND_MAX,
						1.F);
}
