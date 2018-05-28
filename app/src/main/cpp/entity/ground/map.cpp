//
// Created by samuel on 27/05/18.
//

#include "map.h"

Map::Map(glm::vec3 pos, int width, int height, float maxHeight, float *normalizedHeightValues, glm::vec3 scale) {

    heightMap = new HeightMap(normalizedHeightValues, width, height);


    collisionShape = /*new btHeightfieldTerrainShape(width,
                                                   height,
                                                   normalizedHeightValues,
                                                   maxHeight,
                                                   1,
                                                   true,
                                                   false);*/
            new btHeightfieldTerrainShape (width, height,
                normalizedHeightValues, maxHeight, 0.f, maxHeight,
                1, PHY_FLOAT, false);
    collisionShape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));

    this->scale = scale;

    myTransform.setIdentity();
    myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));

    btVector3 intertie(0.f, 0.f, 0.f);

    defaultMotionState = new btDefaultMotionState(myTransform);

    btRigidBody::btRigidBodyConstructionInfo constrInfo(0.f,
                                                        defaultMotionState,
                                                        collisionShape,
                                                        intertie);

    rigidBody = new btRigidBody(constrInfo);
}

void Map::draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) {
    std::tuple<glm::mat4, glm::mat4> matrixes = getMatrixes(pMatrix, vMatrix);
    heightMap->draw(std::get<0>(matrixes), std::get<1>(matrixes), lighPos);
}
