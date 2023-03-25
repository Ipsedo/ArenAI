//
// Created by samuel on 29/08/18.
//

#ifndef PHYVR_CIBLE_H
#define PHYVR_CIBLE_H

#include "../entity/convex.h"
#include "glm/vec3.hpp"

class SupportCible : public Convex {
public:
  SupportCible(AAssetManager *mgr, const glm::vec3 &pos);

  glm::vec3 getPosHinge();
};

class Cible : public Convex {
public:
  Cible(AAssetManager *mgr, SupportCible *supportCible, btDynamicsWorld *world);

  bool isWon();

private:
  btHingeConstraint *hinge;
};

#endif // PHYVR_CIBLE_H
