//
// Created by samuel on 20/03/2023.
//

#include "./joystick.h"

#include <android/input.h>

#include "../utils/logging.h"
#include "../utils/units.h"

HUDJoyStick::HUDJoyStick(AConfiguration *config, int center_x_dp,
                         int center_y_dp, int size_dp, int stick_size_dp)
    : pointer_id(-1), touched(false),
      center_pos_x(dp_to_px(config, center_x_dp)),
      center_pos_y(dp_to_px(config, center_y_dp)),
      stick_x(dp_to_px(config, center_x_dp)),
      stick_y(dp_to_px(config, center_y_dp)), x_value(0.f), y_value(0.f),
      size(dp_to_px(config, size_dp)),
      stick_size(dp_to_px(config, stick_size_dp)) {}

bool HUDJoyStick::on_event(AInputEvent *event) {
  int action = AMotionEvent_getAction(event);
  int pointer_index = (action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >>
                      AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
  int curr_pointer_id = AMotionEvent_getPointerId(event, pointer_index);

  float pt_x = AMotionEvent_getX(event, pointer_index);
  float pt_y = 720 - AMotionEvent_getY(event, pointer_index);

  LOG_INFO("%f %f %d", pt_x, pt_y, is_inside_(pt_x, pt_y));

  switch (AInputEvent_getType(event)) {
  case AMOTION_EVENT_ACTION_DOWN:
    if (!touched && is_inside_(pt_x, pt_y)) {
      pointer_id = curr_pointer_id;
      touched = true;

      stick_x = pt_x;
      stick_y = pt_y;

      return true;
    }
    break;
  case AMOTION_EVENT_ACTION_UP:
    if (touched && pointer_id == curr_pointer_id) {
      touched = false;
      pointer_id = -1;

      stick_x = center_pos_x;
      stick_y = center_pos_y;

      return true;
    }
    break;
  case AMOTION_EVENT_ACTION_MOVE:
    if (touched && pointer_id == curr_pointer_id) {
      float rel_x = pt_x - center_pos_x;
      float rel_y = pt_y - center_pos_y;

      float max_size = (size - stick_size) / 2.f;

      rel_x = rel_x > max_size ? max_size : rel_x;
      rel_y = rel_y > max_size ? max_size : rel_y;

      stick_x = rel_x + center_pos_x;
      stick_y = rel_y + center_pos_y;

      x_value = rel_x / max_size;
      y_value = rel_y / max_size;

      return true;
    }
    break;
  }
  return false;
}

bool HUDJoyStick::is_inside_(float x, float y) const {
  LOG_INFO("%f %f %f %f %f", x, y, center_pos_x, center_pos_y,
           center_pos_x - size / 2.f);

  return (center_pos_x - size / 2.f > x) && (x < center_pos_x + size / 2.f) &&
         (center_pos_y - size / 2.f > y) && (y < center_pos_y + size / 2.f);
}

joystick HUDJoyStick::get_input() { return {x_value, y_value}; }

joystick HUDJoyStick::get_input_px() { return {stick_x, stick_y}; }

std::unique_ptr<HUDDrawable> HUDJoyStick::get_hud_drawable(AAssetManager *mgr) {
  return std::move(std::unique_ptr<HUDDrawable>(new JoyStickDrawable(
      mgr, [this] { return get_input_px(); }, {center_pos_x, center_pos_y},
      size, stick_size)));
}
