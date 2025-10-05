//
// Created by samuel on 28/09/2025.
//

#ifndef PHYVR_CONTROLLER_ENGINE_H
#define PHYVR_CONTROLLER_ENGINE_H

#include <memory>
#include <vector>

#include <android/configuration.h>
#include <android/input.h>

#include <phyvr_controller/controller.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/hud.h>

#include "./button.h"
#include "./joystick.h"

class ControllerEngine {
public:
  ControllerEngine(AConfiguration *config, int width, int height);

  void add_controller(const std::shared_ptr<Controller> &controller);

  int32_t on_event(AInputEvent *event);

  std::vector<std::unique_ptr<HUDDrawable>>
  get_hud_drawables(const std::shared_ptr<AbstractFileReader> &file_reader);

private:
  std::vector<std::shared_ptr<Controller>> controllers;

  std::shared_ptr<HUDJoyStick> drive_joystick;
  std::shared_ptr<HUDButton> fire_button;
  std::shared_ptr<ScreenJoyStick> turret_joystick;
};

#endif// PHYVR_CONTROLLER_ENGINE_H
