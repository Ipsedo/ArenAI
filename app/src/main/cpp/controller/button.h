//
// Created by samuel on 26/03/2023.
//

#ifndef PHYVR_BUTTON_H
#define PHYVR_BUTTON_H

#include <glm/glm.hpp>

#include "./controller.h"

class Button : public EventHandler {
public:
  Button(glm::vec2 pos, float radius);
  bool on_event(AInputEvent *event) override;

private:
  int x, y;
  float radius;
};

#endif // PHYVR_BUTTON_H
