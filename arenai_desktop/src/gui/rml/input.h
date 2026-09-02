//
// Created by samuel on 02/08/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_INPUT_H
#define ARENAI_DESKTOP_GUI_RML_INPUT_H

#include <chrono>
#include <functional>
#include <utility>
#include <variant>
#include <vector>

#include <RmlUi/Core.h>

#include <arenai_controller/callback.h>

#include "../../controller/bindings.h"

namespace arenai::desktop::gui {

    using RawMenuInput = std::variant<
        controller::Key, controller::MouseButton, controller::GamepadButton,
        std::pair<GamepadAxis, double>>;

    class ExplorerNavListener final : public Rml::EventListener {
    public:
        static std::vector<Rml::Element *> visible_file_entries(const Rml::Element *file_list);

        void ProcessEvent(Rml::Event &event) override;
    };

    class MenuInputAdapter final : public controller::AbstractKeyboardCallback,
                                   public controller::AbstractGamepadCallback {
    public:
        MenuInputAdapter(
            Rml::Context *context, std::function<void()> on_escape,
            std::function<void(bool)> on_gamepad_nav);

        void set_capture_sink(std::function<void(const RawMenuInput &)> sink);

        void on_key(controller::Key key, controller::InputAction action) override;
        void on_mouse_move(double x, double y) override;
        void
        on_mouse_button(controller::MouseButton button, controller::InputAction action) override;
        void on_scroll(double x_offset, double y_offset) override;

        void on_gamepad_button(
            controller::GamepadButton button, controller::InputAction action) override;
        void on_joystick(double x, double y, controller::GamepadJoystick stick) override;
        // triggers have no menu role outside binding capture
        void on_trigger(double z, controller::GamepadTrigger trigger) override;

    private:
        void to_gamepad_mode();
        void to_mouse_mode();

        void send_key(Rml::Input::KeyIdentifier key);

        void scroll_at_focus(double value);

        struct AxisNav {
            int direction = 0;// -1 / 0 / +1 after hysteresis
            std::chrono::steady_clock::time_point next_repeat;
        };

        void axis_nav(
            AxisNav &nav, double value, Rml::Input::KeyIdentifier negative_key,
            Rml::Input::KeyIdentifier positive_key);

        Rml::Context *context_;
        std::function<void()> on_escape_;
        std::function<void(bool)> on_gamepad_nav_;
        std::function<void(const RawMenuInput &)> capture_sink_;
        bool gamepad_mode_ = false;
        AxisNav x_nav_;
        AxisNav y_nav_;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_INPUT_H
