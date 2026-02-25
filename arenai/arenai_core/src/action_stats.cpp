//
// Created by samuel on 27/10/2025.
//

#include <arenai_core/action_stats.h>

ActionStats::ActionStats() : has_fire_(false), energy_consumed_(0.f) {}

bool ActionStats::has_fire() const { return has_fire_; }

float ActionStats::energy_consumed() const { return energy_consumed_; }

void ActionStats::process_input(const Action &action) {
    has_fire_ = action.fire_button.pressed;
    energy_consumed_ = (std::abs(action.left_joystick.x) + std::abs(action.left_joystick.y)
                        + std::abs(action.right_joystick.x) + std::abs(action.right_joystick.y))
                       / 4.f;
}
