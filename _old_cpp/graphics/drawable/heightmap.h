//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_HEIGHTMAP_H
#define PHYVR_HEIGHTMAP_H

#include "diffusemodel.h"
#include <vector>
#include "glm/glm.hpp"
#include "BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h"

class HeightMap : public DiffuseModel {
public:
	/**
	 * heightValues is width * height float values
	 * @param heightValues
	 * @param width
	 * @param height
	 */
	HeightMap(btHeightfieldTerrainShape *terrain, float maxheight);

	HeightMap(btHeightfieldTerrainShape *terrain, float maxheight, float color[4]);

private:

	std::vector<float> initPlan(btHeightfieldTerrainShape *terrain, float maxheight);
};


#endif //PHYVR_HEIGHTMAP_H
