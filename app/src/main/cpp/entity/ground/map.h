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

class Map : public Base {
public:
	Map(glm::vec3 pos,
		int width, int length, float *normalizedHeightValues,
		glm::vec3 scale);

	tuple<glm::mat4, glm::mat4> getMatrixes(glm::mat4 pMatrix, glm::mat4 vMatrix) override;

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;

private:
	HeightMap *heightMap;
};


#endif //PHYVR_MAP_H
