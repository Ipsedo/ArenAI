//
// Created by samuel on 29/05/18.
//

#include "cylinder.h"
#include "../../utils/rigidbody.h"
#include "../../utils/assets.h"
#include <glm/gtc/quaternion.hpp>


Cylinder::Cylinder(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
				   DiffuseModel *modelVBO, const glm::vec3 &scale) : Base(constructionInfo, modelVBO,
																			  scale) {

}

Cylinder *
Cylinder::MakeCylinder(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass) {
	std::string objTxt = getFileText(mgr, "obj/cylinder.obj");

	ModelVBO *modelVBO = new ModelVBO(
			objTxt,
			new float[4]{(float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 1.f});

	btCollisionShape *shape = new btCylinderShape(btVector3(1.f, 1.f, 1.f));
	shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotationMatrix);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	btRigidBody::btRigidBodyConstructionInfo cinfo
		= localCreateInfo(mass, myTransform, shape);
	return new Cylinder(cinfo, modelVBO, scale);
}
