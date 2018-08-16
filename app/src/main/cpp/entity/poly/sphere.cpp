//
// Created by samuel on 29/05/18.
//

#include "sphere.h"
#include "../../utils/assets.h"
#include "../../utils/rigidbody.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

auto l = [](glm::vec3 scale) {
	btCollisionShape *shape = new btSphereShape(1.f);
	shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
	return shape;
};

Sphere::Sphere(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass)
	: Poly(Poly::makeCInfo(l, pos, rotMat, scale, mass), Poly::makeModel(mgr, "obj/sphere.obj"), scale) {}
