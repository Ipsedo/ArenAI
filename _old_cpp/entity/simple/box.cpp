//
// Created by samuel on 26/05/18.
//

#include "box.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "../../graphics/drawable/normalmodel.h"

auto l = [](glm::vec3 scale) {
	return new btBoxShape(btVector3(scale.x, scale.y, scale.z));
};

Box::Box(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass)
		: Poly(l, new NormalMapModel(mgr, "obj/cube.obj", "image/canard.png",
									 "textures/NormalMap.png"), pos, scale, rotMat, mass, true) {}
