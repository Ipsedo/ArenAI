//
// Created by samuel on 29/05/18.
//

#include "cone.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

auto l = [](glm::vec3 scale) {
	btCollisionShape *shape = new btConeShape(1.f, 2.f);
	shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
	return shape;
};

Cone::Cone(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass)
		: Poly(Poly::makeCInfo(l, pos, rotMat, scale, mass), Poly::makeModel(mgr, "obj/cone.obj"), scale) {}

Cone::Cone(DiffuseModel *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass)
		: Poly(Poly::makeCInfo(l, pos, rotMat, scale, mass), modelVBO, scale) {}