//
// Created by samuel on 27/05/18.
//

#include "map.h"
#include "../../utils/rigidbody.h"
#include <glm/gtc/type_ptr.hpp>

class TriangleCallBack : public btTriangleCallback {
public:
	glm::vec3 start;
	glm::vec3 end;

	TriangleCallBack() : start(glm::vec3(0.f)), end(glm::vec3(0.f)) {}

	void processTriangle(btVector3 *triangle, int partid, int triangleindex) override {
		glm::vec3 p1 = glm::vec3(triangle[0].getX(), triangle[0].getY(), triangle[0].getZ());
		glm::vec3 p2 = glm::vec3(triangle[1].getX(), triangle[1].getY(), triangle[1].getZ());
		glm::vec3 p3 = glm::vec3(triangle[2].getX(), triangle[2].getY(), triangle[2].getZ());

		start.x = p1.x < start.x ? p1.x : start.x;
		start.x = p2.x < start.x ? p2.x : start.x;
		start.x = p3.x < start.x ? p3.x : start.x;

		start.y = p1.y < start.y ? p1.y : start.y;
		start.y = p2.y < start.y ? p2.y : start.y;
		start.y = p3.y < start.y ? p3.y : start.y;

		start.z = p1.z < start.z ? p1.z : start.z;
		start.z = p2.z < start.z ? p2.z : start.z;
		start.x = p3.z < start.z ? p3.z : start.z;

		// max position
		end.x = p1.x > end.x ? p1.x : end.x;
		end.x = p2.x > end.x ? p2.x : end.x;
		end.x = p3.x > end.x ? p3.x : end.x;

		end.y = p1.y > end.y ? p1.y : end.y;
		end.y = p2.y > end.y ? p2.y : end.y;
		end.y = p3.y > end.y ? p3.y : end.y;

		end.z = p1.z > end.z ? p1.z : end.z;
		end.z = p2.z > end.z ? p2.z : end.z;
		end.x = p3.z > end.z ? p3.z : end.z;
	}
};

Map::Map(const btRigidBody::btRigidBodyConstructionInfo &constructionInfo, DiffuseModel *modelVBO,
		 const glm::vec3 &scale, float* normalizedHeightValues) : Base(constructionInfo, modelVBO, scale, true),
																  normalizedHeightValues(normalizedHeightValues) {

}

Map *makeMap(float *normalizedHeightValues, int width, int length, btVector3 pos, btVector3 scale) {
	btHeightfieldTerrainShape *tmp =
			new btHeightfieldTerrainShape(width, length,
										  normalizedHeightValues, 1.f, 0.f, 1.f,
										  1, PHY_FLOAT, false);
	tmp->setLocalScaling(btVector3(scale));
	HeightMap *heightMap = new HeightMap(tmp, 1.f);

	/*TriangleCallBack *callBack = new TriangleCallBack();
	tmp->processAllTriangles(callBack, btVector3(-1000, -1000, -1000), btVector3(1000, 1000, 1000));
	glm::vec3 minPos = callBack->start;
	glm::vec3 maxPos = callBack->end;
	delete callBack;*/

	btTransform myTransform;
	myTransform.setIdentity();
	myTransform.setOrigin(pos);

	btVector3 intertie(0.f, 0.f, 0.f);
	btDefaultMotionState *motionState = new btDefaultMotionState(myTransform);
	btRigidBody::btRigidBodyConstructionInfo constrInfo(0.f, motionState, tmp, intertie);
	return new Map(constrInfo, heightMap, glm::vec3(1.f), normalizedHeightValues);
}

glm::vec3 Map::getMinPos() {
	return minPos;
}

glm::vec3 Map::getMaxPos() {
	return maxPos;
}

Map::~Map() {
	delete normalizedHeightValues;
}