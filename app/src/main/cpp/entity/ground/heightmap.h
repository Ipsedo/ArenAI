//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_HEIGHTMAP_H
#define PHYVR_HEIGHTMAP_H

#include "../base.h"

#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>

class HeightMap : public Base {
public:
    HeightMap(glm::vec3 pos,
              int heightStickWidth,
              int heightStickLength,
              float* normalizedHeightValues);
};


#endif //PHYVR_HEIGHTMAP_H
