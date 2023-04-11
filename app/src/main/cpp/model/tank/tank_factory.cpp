//
// Created by samuel on 02/04/2023.
//

#include "./tank_factory.h"

template <class... Args>
std::shared_ptr<WheelItem> make_wheel_(bool front_wheel, Args... args) {
  if (front_wheel)
    return std::make_shared<DirectionalWheelItem>(args...);
  else
    return std::make_shared<WheelItem>(args...);
}

TankFactory::TankFactory(AAssetManager *mgr, glm::vec3 chassis_pos)
    : camera(std::nullptr_t()), items(), controllers() {

  glm::vec3 scale(0.5);

  auto chassis_item =
      std::make_shared<ChassisItem>(mgr, chassis_pos, scale, 2000.f);

  camera = chassis_item;
  items.push_back(chassis_item);

  std::vector<std::tuple<std::string, bool, glm::vec3>> wheel_config{
      {"dir_wheel_right_1", true, {-2.7, -1., 3.}},
      {"dir_wheel_left_1", true, {2.7, -1., 3.}},
      {"wheel_right_2", false, {-2.7, -1., 0.}},
      {"wheel_left_2", false, {2.7, -1., 0.}},
      {"wheel_right_3", false, {-2.7, -1., -3.}},
      {"wheel_left_3", false, {2.7, -1., -3.}}};

  for (auto &[wheel_name, is_directional, wheel_pos] : wheel_config) {
    std::shared_ptr<WheelItem> wheel;
    float wheel_mass = 10.f;
    glm::vec3 wheel_scale = scale * glm::vec3(1.2);

    wheel = make_wheel_(is_directional, wheel_name, mgr,
                        wheel_pos + chassis_pos, wheel_pos, wheel_scale,
                        wheel_mass, chassis_item->get_body());

    items.push_back(wheel);
    controllers.push_back(wheel);
  }
}

std::shared_ptr<Camera> TankFactory::get_camera() { return camera; }

std::vector<std::shared_ptr<Item>> TankFactory::get_items() { return items; }

std::vector<std::shared_ptr<Controller>> TankFactory::get_controllers() {
  return controllers;
}
