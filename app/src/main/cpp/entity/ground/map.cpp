//
// Created by samuel on 27/05/18.
//

#include "map.h"
#include "../../utils/rigidbody.h"
#include <glm/gtc/type_ptr.hpp>

/*
Map::Map(glm::vec3 pos, int width, int length, float *normalizedHeightValues, glm::vec3 scale) {
	btHeightfieldTerrainShape *tmp =
			new btHeightfieldTerrainShape(width, length,
										  normalizedHeightValues, 1.f, 0.f, 1.f,
										  1, PHY_FLOAT, false);

	tmp->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
	heightMap = new HeightMap(tmp, 1.f);
	collisionShape.push_back(tmp);

	this->scale.push_back(scale);

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));

	btVector3 intertie(0.f, 0.f, 0.f);

	defaultMotionState.push_back(new btDefaultMotionState(myTransform));

	btRigidBody::btRigidBodyConstructionInfo constrInfo(0.f,
														defaultMotionState[0],
														collisionShape[0],
														intertie);

	rigidBody.push_back(new btRigidBodyWithBase(constrInfo, this));
}*/

Map::Map(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo,
		 btDefaultMotionState *motionState, DiffuseModel *modelVBO,
		 const glm::vec3 &scale) : Base(constructionInfo, motionState, modelVBO, scale) {

}

Map *makeMap(float *normalizedHeightValues, int width, int length, btVector3 pos, btVector3 scale) {
	btHeightfieldTerrainShape *tmp =
			new btHeightfieldTerrainShape(width, length,
										  normalizedHeightValues, 1.f, 0.f, 1.f,
										  1, PHY_FLOAT, false);

	tmp->setLocalScaling(btVector3(scale));
	HeightMap *heightMap = new HeightMap(tmp, 1.f);

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(pos);

	btVector3 intertie(0.f, 0.f, 0.f);
	btDefaultMotionState *motionState = new btDefaultMotionState(myTransform);
	btRigidBody::btRigidBodyConstructionInfo constrInfo(0.f, motionState, tmp, intertie);
	return new Map(constrInfo, motionState, heightMap, glm::vec3(1.f));
}
