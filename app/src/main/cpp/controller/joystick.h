//
// Created by samuel on 20/03/2023.
//

#ifndef PHYVR_JOYSTICK_H
#define PHYVR_JOYSTICK_H

#include <android/configuration.h>
#include <glm/glm.hpp>
#include <memory>

#include "../view/hud.h"
#include "./event.h"
#include "./inputs.h"

class JoyStick : public EventHandler {
public:
  virtual joystick get_input() = 0;
};

class HUDJoyStick : public JoyStick {
public:
  HUDJoyStick(AConfiguration *config, int center_x_dp, int center_y_dp,
              int size_dp, int stick_size_dp);

  bool on_event(AInputEvent *event) override;

  joystick get_input() override;
  joystick get_input_px();

  std::unique_ptr<HUDDrawable> get_hud_drawable(AAssetManager *mgr);

private:
  int pointer_id;
  bool touched;

  const float size, stick_size;

  const float center_pos_x, center_pos_y;
  float stick_x, stick_y;

  float x_value, y_value;

  bool is_inside_(float x, float y) const;
};

#endif // PHYVR_JOYSTICK_H
