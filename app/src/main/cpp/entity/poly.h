//
// Created by samuel on 15/08/18.
//

#ifndef PHYVR_POLY_H
#define PHYVR_POLY_H

#include "../utils/assets.h"
#include "../utils/rigidbody.h"
#include "base.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class Poly : public Base {
private:
	template<typename FunShape>
	static btRigidBodyConstructionInfo makeCInfo(FunShape makeShapFun, glm::vec3 pos,
												 glm::mat4 rotMat, glm::vec3 scale, float mass) {
		btCollisionShape *shape = makeShapFun(scale);

		btTransform myTransform;
		myTransform.setIdentity();
		myTransform.setOrigin(btVector3(pos.x, pos.y, pos.z));
		glm::quat tmp = glm::quat_cast(rotMat);
		myTransform.setRotation(btQuaternion(tmp.x, tmp.y, tmp.z, tmp.w));

		return localCreateInfo(mass, myTransform, shape);
	};

public:
	template<typename FunShape>
	Poly(FunShape makeShapFun, GLDrawable *modelVBO, const glm::vec3 pos,
		 const glm::vec3 &scale, glm::mat4 rotMat, float mass, bool hasOwnModel) : Base(
			makeCInfo(makeShapFun, pos, rotMat, scale, mass), modelVBO, scale, hasOwnModel) {}

	static DiffuseModel *makeModel(AAssetManager *mgr, string objFileName) {
		string objTxt = getFileText(mgr, objFileName);
		return new ModelVBO(objTxt,
							(float) rand() / RAND_MAX,
							(float) rand() / RAND_MAX,
							(float) rand() / RAND_MAX,
							1.f);
	};
};

#endif //PHYVR_POLY_H
