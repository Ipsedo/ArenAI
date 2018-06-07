//
// Created by samuel on 29/05/18.
//

#include "cylinder.h"
#include "../../utils/rigidbody.h"
#include "../../utils/assets.h"
#include <glm/gtc/quaternion.hpp>

Cylinder::Cylinder(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix,
				   float mass) {
	std::string objTxt = getFileText(mgr, "obj/cylinder.obj");

	modelVBO = new ModelVBO(
			objTxt,
			new float[4]{(float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 1.f});

	this->scale.push_back(scale);

	collisionShape.push_back(new btCylinderShape(btVector3(1.f, 1.f, 1.f)));
	collisionShape[0]->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotationMatrix);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	std::tuple<btRigidBody*, btDefaultMotionState*> t = localCreateRigidBody(mass, myTransform, collisionShape[0], this);
	rigidBody.push_back(std::get<0>(t));
	defaultMotionState.push_back(std::get<1>(t));
}

void Cylinder::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	std::tuple<glm::mat4, glm::mat4> matrixes = getMatrixes(pMatrix, vMatrix);
	modelVBO->draw(std::get<0>(matrixes), std::get<1>(matrixes), lighPos);
}

Cylinder::~Cylinder() {
	delete modelVBO;
}
