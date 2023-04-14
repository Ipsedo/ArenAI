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
  std::vector<std::shared_ptr<ItemProducer>> get_item_producers();
  std::vector<std::shared_ptr<Controller>> get_controllers();

  std::map<std::string, std::shared_ptr<Shape>> load_ammu_shapes();

private:
  std::shared_ptr<Camera> camera;
  std::vector<std::shared_ptr<Item>> items;
  std::vector<std::shared_ptr<ItemProducer>> item_producers;
  std::vector<std::shared_ptr<Controller>> controllers;

  AAssetManager *mgr;
};

#endif // PHYVR_TANK_FACTORY_H
