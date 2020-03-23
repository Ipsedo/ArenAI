//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_MAP_H
#define PHYVR_MAP_H

#include "../base.h"
#include "../../graphics/drawable/heightmap.h"

#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>

btRigidBody::btRigidBodyConstructionInfo
makeMapCInfo(btHeightfieldTerrainShape *map, btVector3 pos, btVector3 scale);

DiffuseModel *makeMapModel(btHeightfieldTerrainShape *map);

class Ground : public Base {
public:
	Ground(btHeightfieldTerrainShape *shape, btVector3 pos, btVector3 scale)
			: Base(makeMapCInfo(shape, pos, scale), makeMapModel(shape), glm::vec3(1.0f), true) {};
};

class Map : public Ground {
private:
	glm::vec3 minPos;
	glm::vec3 maxPos;
	float *normalizedHeightValues;

public:
	Map(float *normalizedHeightValues, int width, int length, btVector3 pos, btVector3 scale);

	glm::vec3 getMinPos();

	glm::vec3 getMaxPos();

	~Map();
};

#endif //PHYVR_MAP_H
