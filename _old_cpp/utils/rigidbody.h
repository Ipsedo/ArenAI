//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_RIGODBODY_H
#define PHYVR_RIGODBODY_H

#include <tuple>
#include "glm/glm.hpp"
#include "btBulletDynamicsCommon.h"
#include <string>
#include "glm/gtc/quaternion.hpp"
#include <android/asset_manager.h>

using namespace std;

btRigidBody::btRigidBodyConstructionInfo
localCreateInfo(btScalar mass, const btTransform &startTransform, btCollisionShape *shape);

btConvexHullShape *parseObj(string objFileText);

#endif //PHYVR_RIGODBODY_H
