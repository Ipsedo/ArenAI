//
// Created by samuel on 26/03/2023.
//

#include "./button.h"

Button::Button(glm::vec2 pos, float radius)
    : x(int(pos.x)), y(int(pos.y)), radius(radius) {}

bool Button::on_event(AInputEvent *event) { return false; }
