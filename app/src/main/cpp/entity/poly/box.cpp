//
// Created by samuel on 26/05/18.
//

#include "box.h"
#include "cpp/utils/assets.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

Box::Box(AAssetManager *mgr,
		 glm::vec3 pos,
		 glm::vec3 sideScale,
		 glm::mat4 rotationMatrix,
		 float mass) {

	std::string objTxt = getFileText(mgr, "obj/cube.obj");

	modelVBO = new ModelVBO(
			objTxt,
			new float[4]{(float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 (float) rand() / RAND_MAX,
						 1.f});

	scale.push_back(sideScale);

	collisionShape.push_back(new btBoxShape(btVector3(scale[0].x, scale[0].y, scale[0].z)));

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
	glm::quat tmp = glm::quat_cast(rotationMatrix);
	myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

	btVector3 intertie(0.f, 0.f, 0.f);
	if (mass)
		collisionShape[0]->calculateLocalInertia(mass, intertie);

	defaultMotionState.push_back(new btDefaultMotionState(myTransform));

	btRigidBody::btRigidBodyConstructionInfo constrInfo(mass,
														defaultMotionState[0],
														collisionShape[0],
														intertie);

	rigidBody.push_back(new btRigidBody(constrInfo));
}

void Box::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	std::tuple<glm::mat4, glm::mat4> matrixes = getMatrixes(pMatrix, vMatrix);
	modelVBO->draw(std::get<0>(matrixes), std::get<1>(matrixes), lighPos);
}

Box::~Box() {
	delete modelVBO;
}
