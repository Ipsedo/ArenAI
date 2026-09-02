//
// Created by samuel on 02/08/2026.
//

#include "./input.h"

#include <cmath>
#include <utility>

namespace arenai::desktop::gui {

    std::vector<Rml::Element *>
    ExplorerNavListener::visible_file_entries(const Rml::Element *file_list) {
        std::vector<Rml::Element *> entries;
        for (int i = 0; i < file_list->GetNumChildren(); i++)
            if (Rml::Element *child = file_list->GetChild(i); child->IsVisible())
                entries.push_back(child);
        return entries;
    }

    void ExplorerNavListener::ProcessEvent(Rml::Event &event) {
        const auto key = static_cast<Rml::Input::KeyIdentifier>(
            event.GetParameter<int>("key_identifier", Rml::Input::KI_UNKNOWN));
        if (key != Rml::Input::KI_UP && key != Rml::Input::KI_DOWN) return;

        const Rml::Element *focus = event.GetTargetElement();
        Rml::ElementDocument *document = focus->GetOwnerDocument();
        if (document == nullptr) return;
        const Rml::Element *list = document->GetElementById("file-list");
        Rml::Element *above = document->GetElementById("graphics-configure");
        Rml::Element *below = document->GetElementById("use-folder");
        if (list == nullptr || above == nullptr || below == nullptr) return;

        const auto entries = visible_file_entries(list);
        if (entries.empty()) return;

        Rml::Element *target = nullptr;
        if (focus->GetParentNode() == list) {
            if (key == Rml::Input::KI_UP && focus == entries.front()) target = above;
            else if (key == Rml::Input::KI_DOWN && focus == entries.back()) target = below;
        } else if (key == Rml::Input::KI_DOWN && focus == above) target = entries.front();
        else if (key == Rml::Input::KI_UP && focus == below) target = entries.back();

        if (target != nullptr && target->Focus(true)) {
            target->ScrollIntoView(Rml::ScrollAlignment::Nearest);
            event.StopPropagation();
        }
    }

    MenuInputAdapter::MenuInputAdapter(
        Rml::Context *context, std::function<void()> on_escape,
        std::function<void(bool)> on_gamepad_nav)
        : context_(context), on_escape_(std::move(on_escape)),
          on_gamepad_nav_(std::move(on_gamepad_nav)) {}

    void MenuInputAdapter::set_capture_sink(std::function<void(const RawMenuInput &)> sink) {
        capture_sink_ = std::move(sink);
    }

    void MenuInputAdapter::on_key(const controller::Key key, const controller::InputAction action) {
        if (action != controller::InputAction::Press) return;

        // local copy: the sink uninstalls itself from inside the call
        if (const auto sink = capture_sink_) {
            sink(key);
            return;
        }

        if (key == controller::Key::Escape) on_escape_();
    }

    void MenuInputAdapter::on_mouse_move(const double x, const double y) {
        to_mouse_mode();
        context_->ProcessMouseMove(static_cast<int>(x), static_cast<int>(y), 0);
    }

    void MenuInputAdapter::on_mouse_button(
        const controller::MouseButton button, const controller::InputAction action) {
        // presses feed the capture; releases keep flowing so RmlUi never
        // keeps an element stuck :active
        if (action == controller::InputAction::Press) {
            // local copy: the sink uninstalls itself from inside the call
            if (const auto sink = capture_sink_) {
                sink(button);
                return;
            }
        }

        to_mouse_mode();
        const int index = button == controller::MouseButton::Left    ? 0
                          : button == controller::MouseButton::Right ? 1
                                                                     : 2;
        if (action == controller::InputAction::Press) context_->ProcessMouseButtonDown(index, 0);
        else if (action == controller::InputAction::Release)
            context_->ProcessMouseButtonUp(index, 0);
    }

    void MenuInputAdapter::on_scroll(const double x_offset, const double y_offset) {
        to_mouse_mode();
        // GLFW wheel offsets are positive upwards, RmlUi scrolls
        // positive downwards
        context_->ProcessMouseWheel(
            Rml::Vector2f(-static_cast<float>(x_offset), -static_cast<float>(y_offset)), 0);
    }

    void MenuInputAdapter::on_gamepad_button(
        const controller::GamepadButton button, const controller::InputAction action) {
        if (action != controller::InputAction::Press) return;

        // local copy: the sink uninstalls itself from inside the call
        if (const auto sink = capture_sink_) {
            sink(button);
            return;
        }

        to_gamepad_mode();
        switch (button) {
            case controller::GamepadButton::A: send_key(Rml::Input::KI_RETURN); break;
            case controller::GamepadButton::B: on_escape_(); break;
            case controller::GamepadButton::DPadUp: send_key(Rml::Input::KI_UP); break;
            case controller::GamepadButton::DPadDown: send_key(Rml::Input::KI_DOWN); break;
            case controller::GamepadButton::DPadLeft: send_key(Rml::Input::KI_LEFT); break;
            case controller::GamepadButton::DPadRight: send_key(Rml::Input::KI_RIGHT); break;
            default: break;
        }
    }

    void MenuInputAdapter::on_joystick(
        const double x, const double y, const controller::GamepadJoystick stick) {

        if (const auto sink = capture_sink_) {
            const bool left = stick == controller::GamepadJoystick::Left;
            sink(std::make_pair(left ? GamepadAxis::LeftStickX : GamepadAxis::RightStickX, x));
            if (capture_sink_)
                sink(std::make_pair(left ? GamepadAxis::LeftStickY : GamepadAxis::RightStickY, y));
            return;
        }

        if (stick == controller::GamepadJoystick::Right) {
            scroll_at_focus(y);
            return;
        }
        if (stick != controller::GamepadJoystick::Left) return;

        // GLFW stick y grows downwards, matching KI_DOWN
        axis_nav(x_nav_, x, Rml::Input::KI_LEFT, Rml::Input::KI_RIGHT);
        axis_nav(y_nav_, y, Rml::Input::KI_UP, Rml::Input::KI_DOWN);
    }

    void MenuInputAdapter::on_trigger(const double z, const controller::GamepadTrigger trigger) {
        // local copy: the sink uninstalls itself from inside the call
        const auto sink = capture_sink_;
        if (!sink) return;
        sink(std::make_pair(
            trigger == controller::GamepadTrigger::Left ? GamepadAxis::LeftTrigger
                                                        : GamepadAxis::RightTrigger,
            z));
    }

    void MenuInputAdapter::to_gamepad_mode() {
        if (gamepad_mode_) return;
        gamepad_mode_ = true;

        context_->ProcessMouseLeave();
        on_gamepad_nav_(true);
    }

    void MenuInputAdapter::to_mouse_mode() {
        if (!gamepad_mode_) return;
        gamepad_mode_ = false;
        on_gamepad_nav_(false);
    }

    void MenuInputAdapter::send_key(const Rml::Input::KeyIdentifier key) {
        to_gamepad_mode();
        context_->ProcessKeyDown(key, 0);
        context_->ProcessKeyUp(key, 0);
    }

    void MenuInputAdapter::scroll_at_focus(const double value) {
        constexpr double DEADZONE = 0.3;
        constexpr float SPEED = 10.f;

        if (std::abs(value) < DEADZONE) return;
        to_gamepad_mode();

        Rml::Element *element = context_->GetFocusElement();
        for (; element != nullptr; element = element->GetParentNode())
            if (element->GetComputedValues().overflow_y() != Rml::Style::Overflow::Visible
                && element->GetScrollHeight() > element->GetClientHeight())
                break;
        if (element == nullptr) return;

        // smooth ramp starting at the deadzone edge
        const double amount =
            (value > 0. ? 1. : -1.) * (std::abs(value) - DEADZONE) / (1. - DEADZONE);
        element->SetScrollTop(
            element->GetScrollTop()
            + static_cast<float>(amount) * SPEED * context_->GetDensityIndependentPixelRatio());
    }

    void MenuInputAdapter::axis_nav(
        AxisNav &nav, const double value, const Rml::Input::KeyIdentifier negative_key,
        const Rml::Input::KeyIdentifier positive_key) {
        constexpr double ENGAGE = 0.5, RELEASE = 0.3;
        constexpr auto FIRST_REPEAT = std::chrono::milliseconds(400);
        constexpr auto NEXT_REPEAT = std::chrono::milliseconds(120);

        const int direction = value <= -ENGAGE ? -1 : value >= ENGAGE ? 1 : 0;
        const bool held = nav.direction != 0 && value * nav.direction > RELEASE;
        const auto now = std::chrono::steady_clock::now();

        if (direction != 0 && direction != nav.direction) {
            nav.direction = direction;
            nav.next_repeat = now + FIRST_REPEAT;
            send_key(direction < 0 ? negative_key : positive_key);
        } else if (held && now >= nav.next_repeat) {
            nav.next_repeat = now + NEXT_REPEAT;
            send_key(nav.direction < 0 ? negative_key : positive_key);
        } else if (!held && direction == 0) {
            nav.direction = 0;
        }
    }

}// namespace arenai::desktop::gui
