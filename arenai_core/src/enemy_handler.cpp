//
// Created by samuel on 06/10/2025.
//

#include <arenai_core/enemy_handler.h>

using namespace arenai;
using namespace arenai::core;

namespace arenai::core {

    EnemyControllerHandler::EnemyControllerHandler(
        const float refresh_frequency, const float wanted_fire_frequency,
        const float turret_rad_per_second)
        : nb_frames_to_fire(static_cast<int>(wanted_fire_frequency / refresh_frequency)),
          curr_frame(nb_frames_to_fire),
          turret_scale_per_frame(turret_rad_per_second * refresh_frequency) {}

    std::tuple<bool, controller::user_input> EnemyControllerHandler::to_output(const Action event) {
        curr_frame = std::min(curr_frame + 1, nb_frames_to_fire);

        bool has_fire = false;
        if (event.fire_button.pressed && curr_frame >= nb_frames_to_fire) {
            has_fire = true;
            curr_frame = 0;
        }

        const controller::user_input action{
            .left_joystick = event.left_joystick,
            .right_joystick =
                {.x = event.right_joystick.x * turret_scale_per_frame,
                 .y = event.right_joystick.y * turret_scale_per_frame},
            .fire_button = {has_fire}};

        return {true, action};
    }

}// namespace arenai::core
