//
// Created by samuel on 29/05/18.
//

#ifndef PHYVR_RIGODBODY_H
#define PHYVR_RIGODBODY_H

#include "btBulletDynamicsCommon.h"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include <android/asset_manager.h>
#include <string>
#include <tuple>

using namespace std;

btRigidBody::btRigidBodyConstructionInfo
localCreateInfo(btScalar mass, const btTransform &startTransform,
                btCollisionShape *shape);

btConvexHullShape *parseObj(string objFileText);

#endif // PHYVR_RIGODBODY_H
