//
// Created by samuel on 15/08/18.
//

#ifndef PHYVR_POLY_H
#define PHYVR_POLY_H

#include "base.h"
#include "utils/assets.h"
#include "utils/rigidbody.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

class Poly : public Base {
public:
	Poly(const btRigidBodyConstructionInfo &constructionInfo, DiffuseModel *modelVBO, const glm::vec3 &scale) : Base(
			constructionInfo, modelVBO, scale) {}

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

	static DiffuseModel *makeModel(AAssetManager *mgr, string objFileName) {
		string objTxt = getFileText(mgr, objFileName);
		float* color = new float[4];
		for (int i = 0; i < 3; i++) {
			color[i] = (float) rand() / RAND_MAX;
		}
		color[3] = 1.f;
		return new ModelVBO(objTxt, color);
	};
};

#endif //PHYVR_POLY_H
