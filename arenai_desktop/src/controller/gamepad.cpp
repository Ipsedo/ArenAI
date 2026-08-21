//
// Created by samuel on 15/07/2026.
//

#include "./gamepad.h"

#include <cmath>

namespace arenai::desktop {

    float PlayerGamepadHandler::apply_dead_zone(const double value) {
        constexpr double DEAD_ZONE = 0.05;

        if (std::abs(value) < DEAD_ZONE) return 0.f;

        // ramp from 0 at the deadzone edge up to ±1 at full deflection
        const double sign = value > 0. ? 1. : -1.;
        return static_cast<float>(sign * (std::abs(value) - DEAD_ZONE) / (1. - DEAD_ZONE));
    }

    PlayerGamepadHandler::PlayerGamepadHandler()
        : state{
            .left_stick_x = 0.,
            .left_stick_y = 0.,
            .right_stick_x = 0.,
            .right_stick_y = 0.,
            .left_trigger = 0.,
            .right_trigger = 0.,
            .button = std::nullopt} {}

    void PlayerGamepadHandler::on_gamepad_button(
        const controller::GamepadButton button, const controller::InputAction action) {
        auto event = state;
        event.button = std::make_pair(button, action);
        on_event(event);
    }

    void PlayerGamepadHandler::on_joystick(
        const double x, const double y, const controller::GamepadJoystick stick) {
        switch (stick) {
            case controller::GamepadJoystick::Left:
                state.left_stick_x = x;
                state.left_stick_y = y;
                break;
            case controller::GamepadJoystick::Right:
                state.right_stick_x = x;
                state.right_stick_y = y;

                // the window dispatches the right stick once per frame, after every
                // other gamepad callback: use it as the per-frame tick so the
                // turret / canon deltas are applied exactly once per frame
                on_event(state);
                break;
        }
    }

    void
    PlayerGamepadHandler::on_trigger(const double z, const controller::GamepadTrigger trigger) {
        switch (trigger) {
            case controller::GamepadTrigger::Left: state.left_trigger = z; break;
            case controller::GamepadTrigger::Right: state.right_trigger = z; break;
        }
    }

    std::tuple<bool, controller::user_input>
    PlayerGamepadHandler::to_output(const PlayerGamepadInput event) {
        bool need_fire = false;

        float turret_rotation = 0.f;
        float canon_rotation = 0.f;

        if (event.button.has_value()) {
            if (const auto &[button, action] = event.button.value();
                action == controller::InputAction::Press
                && (button == controller::GamepadButton::RB
                    || button == controller::GamepadButton::LB)) {
                need_fire = true;
            }
        } else {
            // per-frame tick: controllers consume rad/frame deltas, so the stick
            // deflection is scaled into radians here (like the mouse handler)
            constexpr float factor = 0.02f * static_cast<float>(M_PI);

            turret_rotation = factor * apply_dead_zone(event.right_stick_x);
            canon_rotation = factor * apply_dead_zone(event.right_stick_y);
        }

        const float direction = apply_dead_zone(event.left_stick_x);
        // triggers drive the tank: right forward, left backward (both in [0, 1])
        const auto speed = static_cast<float>(event.right_trigger - event.left_trigger);

        return {
            true,
            {.left_joystick = {.x = direction, .y = speed},
             .right_joystick = {.x = turret_rotation, .y = canon_rotation},
             .fire_button = {need_fire}}};
    }

}// namespace arenai::desktop
