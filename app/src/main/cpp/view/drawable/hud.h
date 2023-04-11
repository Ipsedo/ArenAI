//
// Created by samuel on 26/03/2023.
//

#ifndef PHYVR_HUD_H
#define PHYVR_HUD_H

#include <memory>

#include "../../controller/inputs.h"
#include "../program.h"

class HUDDrawable {
public:
  virtual void draw(int width, int height) = 0;
  virtual ~HUDDrawable();
};

class JoyStickDrawable : public HUDDrawable {
public:
  JoyStickDrawable(AAssetManager *mgr,
                   std::function<joystick(void)> get_input_px,
                   glm::vec2 center_px, float size_px, float stick_size_px);

  void draw(int width, int height) override;

  ~JoyStickDrawable() override;

private:
  std::function<joystick(void)> get_input;

  std::unique_ptr<Program> program;

  float center_x, center_y;
  float size, stick_size;
};

#endif // PHYVR_HUD_H
