//
// Created by samuel on 29/05/18.
//

#include "cone.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

 Cone::Cone(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
			DiffuseModel *modelVBO, const glm::vec3 &scale) : Base(constructionInfo, modelVBO, scale) {

 }

Cone *Cone::MakeCone(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass) {
	std::string objTxt = getFileText(mgr, "obj/cone.obj");
	ModelVBO *modelVBO = new ModelVBO(
			objTxt,
			new float[4]{(float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 1.f});
	btRigidBody::btRigidBodyConstructionInfo cinfo = init(pos, scale, rotMat, mass);
	return new Cone(cinfo, modelVBO, scale);
}

Cone *Cone::MakeCone(DiffuseModel *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass) {
	btRigidBody::btRigidBodyConstructionInfo cinfo = init(pos, scale, rotMat, mass);
	return new Cone(cinfo, modelVBO, scale);
}

btRigidBody::btRigidBodyConstructionInfo
Cone::init(glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass) {
	btCollisionShape *shape = new btConeShape(1.f, 2.f);
	shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotationMatrix);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	return localCreateInfo(mass, myTransform, shape);
}
