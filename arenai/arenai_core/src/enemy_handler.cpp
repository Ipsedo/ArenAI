//
// Created by samuel on 06/10/2025.
//

#include <arenai_core/enemy_handler.h>

EnemyControllerHandler::EnemyControllerHandler(
    const float refresh_frequency, const float wanted_fire_frequency,
    const std::shared_ptr<ActionStats> &action_stats, const float turret_rad_per_second)
    : nb_frames_to_fire(static_cast<int>(wanted_fire_frequency / refresh_frequency)),
      curr_frame(nb_frames_to_fire), action_stats(action_stats),
      turret_scale_per_frame(turret_rad_per_second * refresh_frequency) {}

std::tuple<bool, user_input> EnemyControllerHandler::to_output(const Action event) {
    curr_frame = std::min(curr_frame + 1, nb_frames_to_fire);

    bool has_fire = false;
    if (event.fire_button.pressed && curr_frame >= nb_frames_to_fire) {
        has_fire = true;
        curr_frame = 0;
    }

    const user_input action{
        event.left_joystick,
        {event.right_joystick.x * turret_scale_per_frame,
         event.right_joystick.y * turret_scale_per_frame},
        {has_fire}};
    action_stats->process_input(action);

    return {true, action};
}
