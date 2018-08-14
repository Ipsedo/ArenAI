//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_MAP_H
#define PHYVR_MAP_H

#include "entity/base.h"
#include "../../graphics/drawable/heightmap.h"

#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>

class Map : public Base {
public:
	Map(const btRigidBodyConstructionInfo &constructionInfo, btDefaultMotionState *motionState,
		DiffuseModel *modelVBO, const glm::vec3 &scale);
	//HeightMap *heightMap;
};

Map *makeMap(float *normalizedHeightValues, int width, int length, btVector3 pos, btVector3 scale);

#endif //PHYVR_MAP_H
