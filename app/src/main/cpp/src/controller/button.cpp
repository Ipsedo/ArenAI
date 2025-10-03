//
// Created by samuel on 26/03/2023.
//

#include "./button.h"

#include "../units.h"
#include <phyvr_view/hud.h>

HUDButton::HUDButton(AConfiguration *config, int width_px, int height_px,
                     int margin_dp, int size_dp)
    : size(dp_to_px(config, size_dp)),
      center_x(float(width_px) - (dp_to_px(config, margin_dp) + size / 2.f)),
      center_y(dp_to_px(config, margin_dp) + size / 2.f), height(height_px),
      touched(false), pointer_id(-1), pressed(false) {}

bool HUDButton::on_event(AInputEvent *event) {
  int action = AMotionEvent_getAction(event);
  int pointer_index = (action & AMOTION_EVENT_ACTION_POINTER_INDEX_MASK) >>
                      AMOTION_EVENT_ACTION_POINTER_INDEX_SHIFT;
  int curr_pointer_id = AMotionEvent_getPointerId(event, pointer_index);

  float pt_x = AMotionEvent_getX(event, pointer_index);
  float pt_y = float(height) - AMotionEvent_getY(event, pointer_index);

  switch (action & AMOTION_EVENT_ACTION_MASK) {
  case AMOTION_EVENT_ACTION_POINTER_DOWN:
  case AMOTION_EVENT_ACTION_DOWN:
    if (!touched && is_inside_(pt_x, pt_y)) {
      pointer_id = curr_pointer_id;

      touched = true;
      pressed = false;

      return true;
    }
    break;
  case AMOTION_EVENT_ACTION_POINTER_UP:
  case AMOTION_EVENT_ACTION_UP:
    if (touched && pointer_id == curr_pointer_id) {
      touched = false;
      pressed = false;

      pointer_id = -1;

      return true;
    }
    break;
  default:
    break;
  }
  return false;
}

bool HUDButton::is_inside_(float x, float y) const {
  return (center_x - size / 2.f < x) && (x < center_x + size / 2.f) &&
         (center_y - size / 2.f < y) && (y < center_y + size / 2.f);
}

button HUDButton::get_input() {
  if (touched && !pressed) {
    pressed = true;
    return {true};
  }
  return {false};
}

int HUDButton::get_pointer_id() { return pointer_id; }

std::unique_ptr<HUDDrawable> HUDButton::get_hud_drawable(
    const std::shared_ptr<AbstractFileReader> &file_reader) {
  return std::make_unique<ButtonDrawable>(
      file_reader, [this]() { return get_input(); },
      glm::vec2(center_x, center_y), size);
}
