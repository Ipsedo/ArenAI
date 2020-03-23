//
// Created by samuel on 29/05/18.
//

#include "cylinder.h"
#include "../../utils/rigidbody.h"
#include "../../utils/assets.h"
#include <glm/gtc/quaternion.hpp>

auto l = [](glm::vec3 scale) {
	btCollisionShape *shape = new btCylinderShape(btVector3(1.f, 1.f, 1.f));
	shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
	return shape;
};

Cylinder::Cylinder(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix,
				   float mass)
		: Poly(l, Poly::makeModel(mgr, "obj/cylinder.obj"), pos, scale, rotationMatrix, mass,
			   true) {}
