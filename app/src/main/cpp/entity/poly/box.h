//
// Created by samuel on 26/05/18.
//

#ifndef PHYVR_BOX_H
#define PHYVR_BOX_H

#include <btBulletDynamicsCommon.h>
#include <android/asset_manager.h>
#include <glm/glm.hpp>
#include "../../graphics/drawable/modelvbo.h"
#include "entity/base.h"

class Box : public Base {
public:
	static Box *MakeBox(AAssetManager *mgr, glm::vec3 pos, glm::vec3 scale, glm::mat4 rotMat, float mass);

protected:
	Box(const btRigidBodyConstructionInfo &constructionInfo,  DiffuseModel *modelVBO, const glm::vec3 &scale);
};


#endif //PHYVR_CUBE_H
