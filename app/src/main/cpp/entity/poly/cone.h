//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_CONE_H
#define PHYVR_CONE_H


#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "entity/base.h"

class Cone : public Base {
public:
	/*Cone(ModelVBO *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);

	Cone(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;*/

protected:
	Cone(const btRigidBodyConstructionInfo &constructionInfo, btDefaultMotionState *motionState,
		 DiffuseModel *modelVBO, const glm::vec3 &scale);

public:
	static Cone *MakeCone(AAssetManager* mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);
	static Cone *MakeCone(DiffuseModel *modelVBO, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);

private:
	static tuple<btRigidBody::btRigidBodyConstructionInfo, btDefaultMotionState *>
		init(glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);
	/*bool hasOwnModelVBO;

	void init(glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);

	ModelVBO *modelVBO;*/
};


#endif //PHYVR_CONE_H
