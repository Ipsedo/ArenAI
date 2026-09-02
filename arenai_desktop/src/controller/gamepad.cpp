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
                // other gamepad callback: use it as the per-frame tick so the
                // turret / canon deltas are applied exactly once per frame
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

        float turret_rotation = 0.f;
        float canon_rotation = 0.f;

        if (event.button.has_value()) {
            if (const auto &[button, action] = event.button.value();
                action == controller::InputAction::Press && button == bindings.fire)
                need_fire = true;
        } else {
            // per-frame tick: controllers consume rad/frame deltas, so the stick
            // deflection is scaled into radians here (like the mouse handler)
            constexpr float factor = 0.02f * static_cast<float>(M_PI);

            turret_rotation = factor * axis_value(bindings.aim_x, event);
            canon_rotation = factor * axis_value(bindings.aim_y, event);
        }

        const float direction = axis_value(bindings.steer, event);
        // one-way pair driving the tank: accelerate forward, reverse backward
        const float speed = one_way_axis_value(bindings.accelerate, event)
                            - one_way_axis_value(bindings.reverse, event);

        return {
            true,
            {.left_joystick = {.x = direction, .y = speed},
             .right_joystick = {.x = turret_rotation, .y = canon_rotation},
             .fire_button = {need_fire}}};
    }

}// namespace arenai::desktop
