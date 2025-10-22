//
// Created by samuel on 20/10/2025.
//

#include <arenai_core/action_stats.h>

ActionStats::ActionStats(
    const float nb_seconds_idle_tolerated, const float wanted_frequency, const float delta_action)
    : nb_frames_idle(static_cast<int>(nb_seconds_idle_tolerated / wanted_frequency)),
      delta_action(delta_action), frames_since_last_action(0),
      max_frames_penalty(static_cast<int>(10.f / wanted_frequency)), last_action() {}

float ActionStats::get_reward() const {
    return frames_since_last_action >= nb_frames_idle
               ? -static_cast<float>(
                     std::min(frames_since_last_action - nb_frames_idle, max_frames_penalty))
                     / static_cast<float>(max_frames_penalty)
               : 0.f;
}

void ActionStats::process_input(const Action &action) {

    const bool has_move_left_x =
        std::abs(action.left_joystick.x - last_action.left_joystick.x) > delta_action;
    const bool has_move_left_y =
        std::abs(action.left_joystick.y - last_action.left_joystick.y) > delta_action;

    const bool has_move_right_x =
        std::abs(action.right_joystick.x - last_action.right_joystick.x) > delta_action;
    const bool has_move_right_y =
        std::abs(action.right_joystick.y - last_action.right_joystick.y) > delta_action;

    const bool has_fire = action.fire_button.pressed != last_action.fire_button.pressed;

    if (has_move_left_x || has_move_left_y || has_move_right_x || has_move_right_y || has_fire)
        frames_since_last_action = 0;
    else frames_since_last_action++;

    last_action = action;
}
