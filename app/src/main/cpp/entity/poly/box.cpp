//
// Created by samuel on 26/05/18.
//

#include "box.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

Box *Box::MakeBox(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass) {
	std::string objTxt = getFileText(mgr, "obj/cube.obj");

	ModelVBO *modelVBO
			= new ModelVBO(objTxt,
			new float[4]{(float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 1.f});
	btCollisionShape *collisionShape = new btBoxShape(btVector3(scale.x, scale.y, scale.z));

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotMat);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	btRigidBody::btRigidBodyConstructionInfo cinfo = localCreateInfo(mass, myTransform, collisionShape);
	return new Box(cinfo, modelVBO, scale);
}

Box::Box(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
		 DiffuseModel *modelVBO, const glm::vec3 &scale) : Base(constructionInfo, modelVBO, scale) {

}
