//
// Created by samuel on 15/08/18.
//

#ifndef PHYVR_POLY_H
#define PHYVR_POLY_H

#include "base.h"
#include "glm/glm.hpp"
#include <android/asset_manager.h>

class Poly : public Base {
private:
	static btRigidBodyConstructionInfo makeCInfo(
			std::function<btCollisionShape *(glm::vec3)> makeShapeFun, glm::vec3 pos,
			glm::mat4 rotMat, glm::vec3 scale, float mass);

public:
	Poly(std::function<btCollisionShape *(glm::vec3)> makeShapeFun,
		 GLDrawable *modelVBO, const glm::vec3 pos,
		 const glm::vec3 &scale, glm::mat4 rotMat, float mass, bool hasOwnModel);

	static DiffuseModel *makeModel(AAssetManager *mgr, string objFileName);
};

#endif //PHYVR_POLY_H
