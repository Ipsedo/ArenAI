//
// Created by samuel on 27/05/18.
//

#include "heightmap.h"
#include <vector>

HeightMap::HeightMap(btHeightfieldTerrainShape* terrain, float maxheight) {
    init();
    bind();
    bindBuffer(initPlan(terrain, maxheight));

    lightCoef = 0.5f;
    distanceCoef = 0.f;

    for (int i = 0; i < 3; i++)
        this->color[i] = (float) rand() / RAND_MAX;
    this->color[3] = 1.f;
}

HeightMap::HeightMap(btHeightfieldTerrainShape* terrain, float maxheight, float color[4]) {
    init();
    bind();
    bindBuffer(initPlan(terrain, maxheight));

    lightCoef = 1;
    distanceCoef = 0;

    for (int i = 0; i < 4; i++) {
        this->color[i] = color[i];
    }
}

class triangleCallBack : public btTriangleCallback {
public:
    triangleCallBack() {
        nbVertex = 0;
    }
    int nbVertex;
    std::vector<float> packedData;
    void processTriangle(btVector3* triangle, int partid, int triangleindex) override {
        glm::vec3 p1 = glm::vec3(triangle[0].getX(),triangle[0].getY(),triangle[0].getZ());
        glm::vec3 p2 = glm::vec3(triangle[1].getX(),triangle[1].getY(),triangle[1].getZ());
        glm::vec3 p3 = glm::vec3(triangle[2].getX(),triangle[2].getY(),triangle[2].getZ());

        glm::vec3 n1 = glm::cross(p1-p2, p3-p2);
        glm::vec3 n2 = glm::cross(p3-p2,p1-p2);

        // y : up axis
        glm::vec3 n = n1.y > 0 ? n1 : n2;

        packedData.push_back(p1.x);
        packedData.push_back(p1.y);
        packedData.push_back(p1.z);
        packedData.push_back(n.x);
        packedData.push_back(n.y);
        packedData.push_back(n.z);

        packedData.push_back(p2.x);
        packedData.push_back(p2.y);
        packedData.push_back(p2.z);
        packedData.push_back(n.x);
        packedData.push_back(n.y);
        packedData.push_back(n.z);

        packedData.push_back(p3.x);
        packedData.push_back(p3.y);
        packedData.push_back(p3.z);
        packedData.push_back(n.x);
        packedData.push_back(n.y);
        packedData.push_back(n.z);

        nbVertex += 3;
    }
};

std::vector<float> HeightMap::initPlan(btHeightfieldTerrainShape* terrain, float maxheight) {
    nbVertex = 0;

    std::vector<float> res;
    triangleCallBack callback;
    // TODO limits aabb
    terrain->processAllTriangles(&callback, btVector3(-1000, -1000, -1000), btVector3(1000, 1000, 1000));
    nbVertex = callback.nbVertex;
    return callback.packedData;
}
