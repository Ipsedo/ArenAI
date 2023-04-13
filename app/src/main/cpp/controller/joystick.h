//
// Created by samuel on 20/03/2023.
//

#ifndef PHYVR_JOYSTICK_H
#define PHYVR_JOYSTICK_H

#include <android/configuration.h>
#include <glm/glm.hpp>
#include <memory>

#include "../view/drawable/hud.h"
#include "./event.h"
#include "./inputs.h"
#include "./view.h"

class JoyStick : public EventHandler {
public:
  virtual joystick get_input() = 0;
};

class HUDJoyStick : public JoyStick, public PointerLocker {
public:
  HUDJoyStick(AConfiguration *config, int width_px, int height_px,
              int margin_dp, int size_dp, int stick_size_dp);

  bool on_event(AInputEvent *event) override;

  joystick get_input() override;
  joystick get_input_px();

  std::unique_ptr<HUDDrawable> get_hud_drawable(AAssetManager *mgr);

  int get_pointer_id() override;

private:
  int pointer_id;
  bool touched;

  const float size, stick_size;

  float center_pos_x, center_pos_y;
  float stick_x, stick_y;

  float x_value, y_value;

  float pointer_rel_x, pointer_rel_y;

  int width, height;

  bool is_inside_(float x, float y) const;
};

class ScreenJoyStick : public JoyStick {
public:
  ScreenJoyStick(int width, int height,
                 std::vector<std::shared_ptr<PointerLocker>> pointer_lockers);
  bool on_event(AInputEvent *event) override;

  joystick get_input() override;

private:
  float width, height;

  int pointer_id;

  bool touched;

  float last_x, last_y;
  float x_value, y_value;

  std::vector<std::shared_ptr<PointerLocker>> pointer_lockers;

  bool is_pointer_free_(int pointer_id);
};

#endif // PHYVR_JOYSTICK_H
