//
// Created by samuel on 15/07/2026.
//

#ifndef ARENAI_GAMEPAD_HANDLER_H
#define ARENAI_GAMEPAD_HANDLER_H

#include <optional>
#include <utility>

#include <arenai_controller/callback.h>
#include <arenai_controller/handler.h>

namespace arenai::desktop {

    struct PlayerGamepadInput {
        double left_stick_x;
        double left_stick_y;

        double right_stick_x;
        double right_stick_y;

        double left_trigger;
        double right_trigger;

        std::optional<std::pair<controller::GamepadButton, controller::InputAction>> button;
    };

    class PlayerGamepadHandler : public controller::EventHandler<PlayerGamepadInput>,
                                 public controller::AbstractGamepadCallback {
    public:
        PlayerGamepadHandler();

        void on_gamepad_button(
            controller::GamepadButton button, controller::InputAction action) override;
        void on_joystick(double x, double y, controller::GamepadJoystick stick) override;
        void on_trigger(double z, controller::GamepadTrigger trigger) override;

    protected:
        std::tuple<bool, controller::user_input> to_output(PlayerGamepadInput event) override;

    private:
        PlayerGamepadInput state;
    };

}// namespace arenai::desktop

#endif//ARENAI_GAMEPAD_HANDLER_H
