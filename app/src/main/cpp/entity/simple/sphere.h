//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_SPHERE_H
#define PHYVR_SPHERE_H


#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "../poly.h"
#include "../../utils/assets.h"

class Sphere : public Poly {
public:
	Sphere(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);

};


#endif //PHYVR_SPHERE_H
