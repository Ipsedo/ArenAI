//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_MAP_H
#define PHYVR_MAP_H

#include "../base.h"
#include "../../graphics/drawable/heightmap.h"
#include "../../../../../../glm/glm/detail/type_mat.hpp"

#include <BulletCollision/CollisionShapes/btHeightfieldTerrainShape.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>

class Map : public Base {
public:
    Map(glm::vec3 pos,
        int width, int height, float maxHeight, float *normalizedHeightValues,
        glm::vec3 scale);

    void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;

private:
    HeightMap* heightMap;
};


#endif //PHYVR_MAP_H
