//
// Created by samuel on 26/05/18.
//

#include "box.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

auto l = [](glm::vec3 scale) {
	return new btBoxShape(btVector3(scale.x, scale.y, scale.z));
};

Box::Box(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass)
		: Poly(l, Poly::makeModel(mgr, "obj/cube.obj"), pos, scale, rotMat, mass, true) {}
