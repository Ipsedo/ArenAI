//
// Created by samuel on 06/10/2025.
//

#include <arenai_core/enemy_handler.h>

EnemyControllerHandler::EnemyControllerHandler(
    const float refresh_frequency, const float wanted_fire_frequency)
    : nb_frames_to_fire(static_cast<int>(wanted_fire_frequency / refresh_frequency)),
      curr_frame(nb_frames_to_fire) {}

std::tuple<bool, user_input> EnemyControllerHandler::to_output(const Action event) {
    curr_frame = std::min(curr_frame + 1, nb_frames_to_fire);

    bool has_fire = false;
    if (event.fire_button.pressed && curr_frame >= nb_frames_to_fire) {
        has_fire = true;
        curr_frame = 0;
    }

    return {true, {event.left_joystick, event.right_joystick, {has_fire}}};
}
