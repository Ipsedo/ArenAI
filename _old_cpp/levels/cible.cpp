//
// Created by samuel on 29/08/18.
//

#include "cible.h"

#include "glm/glm.hpp"

#define CIBLE_SCALE glm::vec3(8.f, 6.f, 4.f)
#define SUPPORT_SCALE 4.f
#define HEIGHT_SUPPORT 0.3f

Cible::Cible(AAssetManager *mgr, SupportCible *supportCible, btDynamicsWorld *world)
    : Convex(
        mgr, "obj/cible.obj", supportCible->getPosHinge() + glm::vec3(0, CIBLE_SCALE.y, 0.f),
        CIBLE_SCALE, glm::mat4(1.f), 1000.f) {

    btRigidBody *pBodyA = supportCible;
    btRigidBody *pBodyB = this;

    btVector3 pivotA = btVector3(0, HEIGHT_SUPPORT, 0);
    btVector3 pivotB = btVector3(0.f, -scale.y, 0.f);
    btVector3 axis = btVector3(1.f, 0.f, 0.f);

    hinge = new btHingeConstraint(*pBodyA, *pBodyB, pivotA, pivotB, axis, axis, true);
    world->addConstraint(hinge, true);
    hinge->setLimit((float) (-M_PI * 1e-1), (float) (M_PI * 0.5));

    setIgnoreCollisionCheck(supportCible, true);
}

bool Cible::isWon() { return hinge->getHingeAngle() > (float) (M_PI * 0.4); }

SupportCible::SupportCible(AAssetManager *mgr, const glm::vec3 &pos)
    : Convex(mgr, "obj/support_cible.obj", pos, glm::vec3(SUPPORT_SCALE), glm::mat4(1.f), 0.f) {}

glm::vec3 SupportCible::getPosHinge() {
    btVector3 pos = getWorldTransform().getOrigin();
    return glm::vec3(pos.x(), pos.y(), pos.z()) + HEIGHT_SUPPORT * glm::vec3(0, scale.y, 0);
}
