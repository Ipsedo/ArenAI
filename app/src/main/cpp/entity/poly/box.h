//
// Created by samuel on 26/05/18.
//

#ifndef PHYVR_BOX_H
#define PHYVR_BOX_H

#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "cpp/graphics/drawable/modelvbo.h"
#include "cpp/entity/base.h"

class Box : public Base {
public:
	Box(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;

	~Box();

private:
	ModelVBO *modelVBO;
};


#endif //PHYVR_CUBE_H
