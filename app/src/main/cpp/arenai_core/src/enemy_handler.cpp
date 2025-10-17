//
// Created by samuel on 06/10/2025.
//

#include <arenai_core/enemy_handler.h>

EnemyControllerHandler::EnemyControllerHandler(const float fire_latency_seconds)
    : fire_latency_seconds(fire_latency_seconds), last_time(std::chrono::steady_clock::now()) {}

std::tuple<bool, user_input> EnemyControllerHandler::to_output(const Action event) {

    const auto now = std::chrono::steady_clock::now();
    const auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(last_time - now);
    const auto can_fire = static_cast<float>(dt.count()) >= 1000.f * fire_latency_seconds;

    const button fire_button_restricted(event.fire_button.pressed && can_fire);

    if (can_fire) last_time = now;

    return {
        true,
        user_input(
            event.left_joystick, event.right_joystick, fire_button_restricted, {false}, {false})};
}
