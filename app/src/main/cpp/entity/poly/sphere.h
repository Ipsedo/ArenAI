//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_SPHERE_H
#define PHYVR_SPHERE_H


#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "entity/base.h"

class Sphere : public Base {
public:
	/*Sphere(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotationMatrix, float mass);

	void draw(glm::mat4 pMatrix, glm::mat4 vMatrix, glm::vec3 lighPos) override;

	~Sphere();*/
	static Sphere *MakeSphere(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);
protected:
	Sphere(const btRigidBodyConstructionInfo &constructionInfo,
		   DiffuseModel *modelVBO, const glm::vec3 &scale);

private:
	//ModelVBO *modelVBO;
};


#endif //PHYVR_SPHERE_H
