//
// Created by samuel on 27/05/18.
//

#include "heightmap.h"

HeightMap::HeightMap(glm::vec3 pos, int heightStickWidth, int heightStickLength, float *normalizedHeightValues) {
    float maxHeight = 10.f;
    collisionShape = new btHeightfieldTerrainShape(heightStickWidth,
                                                   heightStickLength,
                                                   normalizedHeightValues,
                                                   maxHeight,
                                                   1,
                                                   true,
                                                   false);

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
