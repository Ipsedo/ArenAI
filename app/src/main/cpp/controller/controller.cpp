//
// Created by samuel on 19/03/2023.
//

#include "./controller.h"
#include "../utils/logging.h"

#include <android/input.h>
#include <android_native_app_glue.h>

ControllerEngine::ControllerEngine(AConfiguration *config, int width,
                                   int height)
    : controllers() {
  drive_joystick =
      std::make_shared<HUDJoyStick>(config, width, height, 40, 150, 40);
  turret_joystick = std::make_shared<ScreenJoyStick>(
      width, height,
      std::vector<std::shared_ptr<PointerLocker>>{drive_joystick});
}

void ControllerEngine::add_controller(
    const std::shared_ptr<Controller> &controller) {
  controllers.push_back(controller);
}

int32_t ControllerEngine::on_event(AInputEvent *event) {
  int drive_joystick_used = drive_joystick->on_event(event);
  int turret_joystick_used = turret_joystick->on_event(event);

  if (drive_joystick_used || turret_joystick_used) {
    user_input input{drive_joystick->get_input(),
                     turret_joystick->get_input(),
                     {false},
                     {false},
                     {false}};

    for (auto &ctrl : controllers)
      ctrl->on_input(input);

    return 1;
  }

  return 0;
}

std::vector<std::unique_ptr<HUDDrawable>>
ControllerEngine::get_hud_drawables(AAssetManager *mgr) {
  std::vector<std::unique_ptr<HUDDrawable>> result{};

  auto left_joystick_drawable = drive_joystick->get_hud_drawable(mgr);
  result.push_back(std::move(left_joystick_drawable));

  /*
  auto right_joystick_drawable = turret_joystick->get_hud_drawable(mgr);
  result.push_back(std::move(right_joystick_drawable));*/

  return result;
}
