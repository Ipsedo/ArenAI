//
// Created by samuel on 11/06/18.
//

#ifndef PHYVR_LEVELDEMO_H
#define PHYVR_LEVELDEMO_H

#include "../entity/tank/tank.h"
#include "../graphics/drawable/cubemap.h"
#include "level.h"

class LevelDemo : public Level {
public:
  LevelDemo();

  void init(bool isVR, AAssetManager *mgr, btDynamicsWorld *world) override;

  vector<Controls *> getControls() override;

  vector<Shooter *> getShooters() override;

  Camera *getCamera() override;

  vector<Base *> getEntities() override;

  vector<Drawable *> getDrawables() override;

  Limits getLimits() override;

  glm::vec3 getLightPos() override;

  bool won() override;

  bool lose() override;

  void step() override;

  ~LevelDemo();

private:
  CubeMap *map;
  Tank *tank;
  bool isInit;
};

#endif // PHYVR_LEVEL0_H
