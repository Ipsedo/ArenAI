//
// Created by samuel on 02/04/2023.
//

#ifndef PHYVR_TANK_FACTORY_H
#define PHYVR_TANK_FACTORY_H

#include <memory>

#include "./canon.h"
#include "./chassis.h"
#include "./turret.h"
#include "./wheel.h"

#include "../../controller/controller.h"

class TankFactory {
public:
  TankFactory(AAssetManager *mgr, glm::vec3 chassis_pos);
  std::shared_ptr<Camera> get_camera();
  std::vector<std::shared_ptr<Item>> get_items();
  std::vector<std::shared_ptr<Controller>> get_controllers();

private:
  std::shared_ptr<Camera> camera;
  std::vector<std::shared_ptr<Item>> items;
  std::vector<std::shared_ptr<Controller>> controllers;
};

#endif // PHYVR_TANK_FACTORY_H
