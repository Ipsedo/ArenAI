//
// Created by samuel on 27/05/18.
//

#include "map.h"
#include "../../utils/rigidbody.h"
#include <glm/gtc/type_ptr.hpp>

Map::Map(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo, DiffuseModel *modelVBO,
		 const glm::vec3 &scale) : Base(constructionInfo, modelVBO, scale) {

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
	return new Map(constrInfo, heightMap, glm::vec3(1.f));
}
