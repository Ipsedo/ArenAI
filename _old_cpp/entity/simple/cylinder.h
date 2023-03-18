//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_CYLINDER_H
#define PHYVR_CYLINDER_H

#include "../../utils/assets.h"
#include "../poly.h"
#include "glm/glm.hpp"
#include <android/asset_manager.h>

class Cylinder : public Poly {
public:
	Cylinder(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale,
			 glm::mat4 rotationMatrix, float mass);
};


#endif //PHYVR_CYLINDER_H
