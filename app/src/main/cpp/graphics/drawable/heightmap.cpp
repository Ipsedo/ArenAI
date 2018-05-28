//
// Created by samuel on 27/05/18.
//

#include "heightmap.h"
#include <vector>
#include <android/log.h>

HeightMap::HeightMap(float *heightValues, int width, int height) {
    init();
    bind();
    bindBuffer(initPlan(heightValues, width, height));

    lightCoef = 1;
    distanceCoef = 0;

    for (int i = 0; i < 3; i++)
        this->color[i] = (float) rand() / RAND_MAX;
    this->color[3] = 1.f;
}

HeightMap::HeightMap(float *heightValues, int width, int height, float *color) {
    init();
    bind();
    bindBuffer(initPlan(heightValues, width, height));

    lightCoef = 1;
    distanceCoef = 0;

    for (int i = 0; i < 4; i++) {
        this->color[i] = color[i];
    }
}

std::vector<float> HeightMap::initPlan(float* heightValues, int width, int height) {
    nbVertex = 0;
    float maxSide = width > height ? width : height;
    float sideX = 2.f * float(width) / maxSide, sideY = 2.f * float(height) / maxSide;

    float startX = - 0.5f * sideX, deltaX = sideX / float(width);
    float startY = - 0.5f * sideY, deltaY = sideY / float(height);

    float currX = startX, currY = startY;

    // STRIDE = glm::vec3 + glm::vec3
    //          point        normal
    std::vector<glm::vec3> packedDataNormal;
    for (int i = 0; i < height-1; i++) {
        for (int j = 0; j < width-1; j++) {
            /**
             * 1------3-...
             * |      |
             * |      |
             * 2------4-....
             *
             * premier triangle 1-3-2
             * deuxieme triangle 2-3-4
             */
            glm::vec3 p1(currX,
                         heightValues[i * width + j],
                         currY);
            glm::vec3 p2(currX,
                         heightValues[(i + 1) * width + j],
                         currY + deltaY);
            glm::vec3 p3(currX + deltaX,
                         heightValues[i * width + j + 1],
                         currY);
            glm::vec3 p4(currX + deltaX,
                         heightValues[(i + 1) * width + j + 1],
                         currY + deltaY);

            glm::vec3 n1 = -glm::cross(p3-p1, p2-p1);
            glm::vec3 n2 = glm::cross(p2-p3, p4-p3); //Pas bon

            // Triangle 1
            packedDataNormal.push_back(p1);
            packedDataNormal.push_back(n1);

            packedDataNormal.push_back(p2);
            packedDataNormal.push_back(n1);

            packedDataNormal.push_back(p3);
            packedDataNormal.push_back(n1);

            // Triangle 2
            // Pas bon
            packedDataNormal.push_back(p2);
            packedDataNormal.push_back(n2);

            packedDataNormal.push_back(p3);
            packedDataNormal.push_back(n2);

            packedDataNormal.push_back(p4);
            packedDataNormal.push_back(n2);

            currX += deltaX;
            nbVertex += 6;
        }
        currX = startX;
        currY += deltaY;
    }

    std::vector<float> res;
    for (int i = 0; i < packedDataNormal.size(); i++) {
        glm::vec3 curr = packedDataNormal[i];
        res.push_back(curr.x);
        res.push_back(curr.y);
        res.push_back(curr.z);
    }
    return res;
}
