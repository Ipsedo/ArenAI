//
// Created by samuel on 19/03/2023.
//

#include "./controller.h"
#include "../utils/logging.h"

#include <android/input.h>

ControllerEngine::ControllerEngine(AConfiguration *config) {
  left_joystick =
      std::make_shared<HUDJoyStick>(config, 10 + 100, 10 + 100, 600, 25);
}

void ControllerEngine::add_controller(
    const std::string &name, const std::shared_ptr<Controller> &controller) {
  controllers.insert({name, controller});
}

void ControllerEngine::remove_controller(const std::string &name) {
  controllers.erase(name);
}

int32_t ControllerEngine::on_event(AInputEvent *event) {
  left_joystick->on_event(event);
  return 1;
}

std::vector<std::unique_ptr<HUDDrawable>>
ControllerEngine::get_hud_drawables(AAssetManager *mgr) {
  auto left_joystick_drawable = left_joystick->get_hud_drawable(mgr);
  std::vector<std::unique_ptr<HUDDrawable>> result{};
  result.push_back(std::move(left_joystick_drawable));
  return result;
}
