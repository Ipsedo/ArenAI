//
// Created by samuel on 27/05/18.
//

#include "map.h"
#include "../../utils/rigidbody.h"
#include <glm/gtc/type_ptr.hpp>

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
}

/**
 * Mesh is already scaled !
 * @param pMatrix
 * @param vMatrix
 * @return
 */
std::tuple<glm::mat4, glm::mat4> Map::getMatrixes(glm::mat4 pMatrix, glm::mat4 vMatrix) {
	btScalar tmp[16];
	defaultMotionState[0]->m_graphicsWorldTrans.getOpenGLMatrix(tmp);
	glm::mat4 modelMatrix = glm::make_mat4(tmp);

	glm::mat4 mvMatrix = vMatrix * modelMatrix;
	glm::mat4 mvpMatrix = pMatrix * mvMatrix;

	return tuple<glm::mat4, glm::mat4>(mvpMatrix, mvMatrix);
}

void Map::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
	std::tuple<glm::mat4, glm::mat4> matrixes = getMatrixes(pMatrix, vMatrix);
	heightMap->draw(std::get<0>(matrixes), std::get<1>(matrixes), lighPos);
}
