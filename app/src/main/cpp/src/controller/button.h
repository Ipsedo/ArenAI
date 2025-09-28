//
// Created by samuel on 26/03/2023.
//

#ifndef PHYVR_BUTTON_H
#define PHYVR_BUTTON_H

#include <android/asset_manager.h>
#include <android/configuration.h>
#include <glm/glm.hpp>
#include <memory>

#include "./event.h"
#include <phyvr_controller/inputs.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/hud.h>

class Button : public EventHandler {
public:
  virtual button get_input() = 0;
};

class HUDButton : public Button, public PointerLocker {
public:
  HUDButton(AConfiguration *config, int width_px, int height_px, int margin_dp,
            int size_dp);
  bool on_event(AInputEvent *event) override;

  button get_input() override;

  int get_pointer_id() override;

  std::unique_ptr<HUDDrawable>
  get_hud_drawable(const std::shared_ptr<AbstractFileReader> &file_reader);

private:
  float size;
  float center_x, center_y;
  int height;

  bool touched;
  int pointer_id;

  bool pressed;

  bool is_inside_(float x, float y) const;
};

#endif // PHYVR_BUTTON_H
