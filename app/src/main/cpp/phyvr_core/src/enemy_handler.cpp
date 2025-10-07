//
// Created by samuel on 06/10/2025.
//

#include <phyvr_core/enemy_handler.h>

EnemyControllerHandler::EnemyControllerHandler(float fire_latency_seconds)
    : fire_latency_seconds(fire_latency_seconds), last_time(std::chrono::steady_clock::now()) {}

std::tuple<bool, user_input> EnemyControllerHandler::to_output(Action event) {

    auto now = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(last_time - now);
    auto can_fire = static_cast<float>(dt.count()) >= 1000.f * fire_latency_seconds;

    button fire_button_restricted(event.fire_button.pressed && can_fire);

    if (can_fire) last_time = now;

    return {
        true,
        user_input(
            event.left_joystick, event.right_joystick, fire_button_restricted, {false}, {false})};
}
