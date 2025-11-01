//
// Created by samuel on 27/10/2025.
//

#include <arenai_core/action_stats.h>

ActionStats::ActionStats() : has_fire_(false) {}

bool ActionStats::has_fire() const { return has_fire_; }

void ActionStats::process_input(const Action &action) { has_fire_ = action.fire_button.pressed; }
