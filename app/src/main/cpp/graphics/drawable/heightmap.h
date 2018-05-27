//
// Created by samuel on 27/05/18.
//

#ifndef PHYVR_HEIGHTMAP_H
#define PHYVR_HEIGHTMAP_H

#include "diffusemodel.h"
#include <vector>
#include <glm/glm.hpp>

class HeightMap : public DiffuseModel {
public:
    /**
     * heightValues is width * height float values
     * @param heightValues
     * @param width
     * @param height
     */
    HeightMap(float* heightValues, int width, int height);
    HeightMap(float* heightValues, int width, int height, float color[4]);
private:

    std::vector<float> initPlan(float* heightValues, int width, int height);
};


#endif //PHYVR_HEIGHTMAP_H
