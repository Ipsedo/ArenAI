//
// Created by samuel on 15/07/2026.
//

#include "./gamepad.h"

#include <algorithm>
#include <cmath>

namespace arenai::desktop {

    namespace {

        double &axis_slot(PlayerGamepadInput &input, const GamepadAxis axis) {
            return input.axes[static_cast<size_t>(axis)];
        }

        double axis_slot(const PlayerGamepadInput &input, const GamepadAxis axis) {
            return input.axes[static_cast<size_t>(axis)];
        }

    }// namespace

    float PlayerGamepadHandler::apply_dead_zone(const double value) {
        constexpr double DEAD_ZONE = 0.05;

        if (std::abs(value) < DEAD_ZONE) return 0.f;

        // ramp from 0 at the deadzone edge up to ±1 at full deflection
        const double sign = value > 0. ? 1. : -1.;
        return static_cast<float>(sign * (std::abs(value) - DEAD_ZONE) / (1. - DEAD_ZONE));
    }

    float PlayerGamepadHandler::axis_value(
        const std::optional<GamepadAxisBinding> &slot, const PlayerGamepadInput &event) {
        if (!slot) return 0.f;
        return apply_dead_zone(axis_slot(event, slot->axis));
    }

    float PlayerGamepadHandler::one_way_axis_value(
        const std::optional<GamepadAxisBinding> &slot, const PlayerGamepadInput &event) {
        if (!slot) return 0.f;
        return std::max(0.f, slot->sign * apply_dead_zone(axis_slot(event, slot->axis)));
    }

    PlayerGamepadHandler::PlayerGamepadHandler(const GamepadBindings &bindings)
        : bindings(bindings), state{.axes = {}, .button = std::nullopt} {}

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
                axis_slot(state, GamepadAxis::LeftStickX) = x;
                axis_slot(state, GamepadAxis::LeftStickY) = y;
                break;
            case controller::GamepadJoystick::Right:
                axis_slot(state, GamepadAxis::RightStickX) = x;
                axis_slot(state, GamepadAxis::RightStickY) = y;

                // the window dispatches the right stick once per frame, after every
                // other gamepad callback: use it as the per-frame tick so the stick
                // deflection is integrated into the aim exactly once per frame
                on_event(state);
                break;
        }
    }

    void
    PlayerGamepadHandler::on_trigger(const double z, const controller::GamepadTrigger trigger) {
        switch (trigger) {
            case controller::GamepadTrigger::Left:
                axis_slot(state, GamepadAxis::LeftTrigger) = z;
                break;
            case controller::GamepadTrigger::Right:
                axis_slot(state, GamepadAxis::RightTrigger) = z;
                break;
        }
    }

    std::tuple<bool, controller::user_input>
    PlayerGamepadHandler::to_output(const PlayerGamepadInput event) {
        bool need_fire = false;

        if (event.button.has_value()) {
            const auto &[button, action] = event.button.value();
            if (action == controller::InputAction::Press && button == bindings.fire)
                need_fire = true;

            if (button == bindings.zoom) {
                if (action == controller::InputAction::Press) zoom_held = true;
                else if (action == controller::InputAction::Release) zoom_held = false;
            }
        } else {
            // per-frame tick: the stick deflection is a rotation speed, integrated here into
            // the absolute aim the parts consume. the gains are the historical rad/frame
            // factors expressed in the parts' normalized range, so the feel is unchanged
            constexpr float turret_gain = 0.02f;
            constexpr float canon_gain = 0.04f;

            // wrap like the turret, clamp like the canon: no windup past the parts' range
            turret_norm =
                std::remainder(turret_norm - turret_gain * axis_value(bindings.aim_x, event), 2.f);
            canon_norm =
                std::clamp(canon_norm + canon_gain * axis_value(bindings.aim_y, event), -1.f, 1.f);
        }

        // button events carry no aim: they re-emit the held target untouched, zeroing it
        // would snap the turret back to dead ahead on every shot

        const float direction = axis_value(bindings.steer, event);
        // one-way pair driving the tank: accelerate forward, reverse backward
        const float speed = one_way_axis_value(bindings.accelerate, event)
                            - one_way_axis_value(bindings.reverse, event);

        return {
            true,
            {.left_joystick = {.x = direction, .y = speed},
             .right_joystick = {.x = turret_norm, .y = canon_norm},
             .fire_button = {need_fire},
             .zoom_button = {zoom_held}}};
    }

}// namespace arenai::desktop
