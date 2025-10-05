//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_RIGODBODY_H
#define PHYVR_RIGODBODY_H

#include <string>
#include <tuple>

#include <android/asset_manager.h>

#include "btBulletDynamicsCommon.h"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"

using namespace std;

btRigidBody::btRigidBodyConstructionInfo
localCreateInfo(btScalar mass, const btTransform &startTransform, btCollisionShape *shape);

btConvexHullShape *parseObj(string objFileText);

#endif// PHYVR_RIGODBODY_H
