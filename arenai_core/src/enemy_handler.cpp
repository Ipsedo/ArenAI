//
// Created by samuel on 06/10/2025.
//

#include <arenai_core/enemy_handler.h>

using namespace arenai;
using namespace arenai::core;

namespace arenai::core {

    EnemyControllerHandler::EnemyControllerHandler(
        const float refresh_frequency, const float turret_rad_per_second)
        : turret_scale_per_frame(turret_rad_per_second * refresh_frequency) {}

    std::tuple<bool, controller::user_input> EnemyControllerHandler::to_output(const Action event) {
        const controller::user_input action{
            .left_joystick = event.left_joystick,
            .right_joystick =
                {.x = event.right_joystick.x * turret_scale_per_frame,
                 .y = event.right_joystick.y * turret_scale_per_frame},
            .fire_button = event.fire_button};

        return {true, action};
    }

}// namespace arenai::core
