//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_GAMEPAD_HANDLER_H
#define ARENAI_GAMEPAD_HANDLER_H

#include <array>
#include <optional>
#include <utility>

#include <arenai_controller/callback.h>
#include <arenai_controller/handler.h>

#include "./bindings.h"

namespace arenai::desktop {

    struct PlayerGamepadInput {
        // raw values indexed by GamepadAxis: sticks in [-1, 1], triggers
        // normalized to [0, 1] by the window
        std::array<double, NB_GAMEPAD_AXES> axes;

        std::optional<std::pair<controller::GamepadButton, controller::InputAction>> button;
    };

    class PlayerGamepadHandler : public controller::EventHandler<PlayerGamepadInput>,
                                 public controller::AbstractGamepadCallback {
    public:
        explicit PlayerGamepadHandler(const GamepadBindings &bindings);

        void on_gamepad_button(
            controller::GamepadButton button, controller::InputAction action) override;
        void on_joystick(double x, double y, controller::GamepadJoystick stick) override;
        void on_trigger(double z, controller::GamepadTrigger trigger) override;

    protected:
        std::tuple<bool, controller::user_input> to_output(PlayerGamepadInput event) override;

    private:
        GamepadBindings bindings;

        PlayerGamepadInput state;

        static float apply_dead_zone(double value);

        // deflection of a two-way slot (steer, aim), 0 when unbound
        float
        axis_value(const std::optional<GamepadAxisBinding> &slot, const PlayerGamepadInput &event);
        // deflection of a one-way slot (accelerate, reverse): the captured
        // direction reads positive, the other way is ignored
        float one_way_axis_value(
            const std::optional<GamepadAxisBinding> &slot, const PlayerGamepadInput &event);
    };

}// namespace arenai::desktop

#endif//ARENAI_GAMEPAD_HANDLER_H
