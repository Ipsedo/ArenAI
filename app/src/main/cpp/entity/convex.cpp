//
// Created by samuel on 27/05/18.
//

#include "convex.h"

#include <glm/gtc/quaternion.hpp>

#include "../utils/rigidbody.h"
#include "../utils/string_utils.h"
#include "../utils/assets.h"

Convex::Convex(AAssetManager *mgr, string objFileName, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix,
			   float mass)
	: Poly(Poly::makeCInfo([mgr, objFileName](glm::vec3 scale){
		string objTxt = getFileText(mgr, objFileName);
		btCollisionShape *shape = parseObj(objTxt);
		shape->setLocalScaling(btVector3(scale.x, scale.y, scale.z));
		return shape;
	}, pos, rotationMatrix, scale, mass), Poly::makeModel(mgr, objFileName), scale, true)  {}
