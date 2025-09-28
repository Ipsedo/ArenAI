//
// Created by samuel on 28/09/2025.
//

#ifndef PHYVR_TANK_FACTORY_H
#define PHYVR_TANK_FACTORY_H

#include "./item.h"
#include <map>
#include <memory>

#include <phyvr_controller/controller.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/camera.h>

class TankFactory {
public:
  TankFactory(const std::shared_ptr<AbstractFileReader> &file_reader,
              glm::vec3 chassis_pos);

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

  std::shared_ptr<AbstractFileReader> file_reader;
};

#endif // PHYVR_TANK_FACTORY_H
