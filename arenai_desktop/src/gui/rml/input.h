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

    // one raw input event handed to the controls page while it captures a
    // binding: a key, a mouse button, a pad button, or a pad axis deflection
    using RawMenuInput = std::variant<
        controller::Key, controller::MouseButton, controller::GamepadButton,
        std::pair<GamepadAxis, double>>;

    // The entries of the file explorer live inside a scroll container
    // (.file-list, overflow-y: auto), and RmlUi's spatial navigation
    // treats scroll containers as isolated islands: the search never
    // descends into one from outside nor climbs out from inside
    // (SearchNavigationTarget in RmlUi's ElementDocument), so the D-pad
    // alone can neither reach nor leave the list. This keydown listener
    // bridges the island's borders before the document runs its default
    // navigation: Down from the Display toggles dives onto the first
    // entry, Up / Down on an edge entry resurfaces onto the neighbour
    // rows; strictly between entries RmlUi's native navigation already
    // moves the focus and scrolls it into view by itself.
    class ExplorerNavListener final : public Rml::EventListener {
    public:
        // the data-for template stays in the list as a display:none
        // child; only its instantiated clones are real entries
        static std::vector<Rml::Element *> visible_file_entries(const Rml::Element *file_list);

        void ProcessEvent(Rml::Event &event) override;
    };

    // Adapts the window's input port to RmlUi context events: the menus
    // reuse the same controller callbacks as the game, so GLFW types
    // never surface here. The gamepad half rides RmlUi's keyboard
    // navigation: A activates the focused element (Enter), B backs out
    // (the Escape path), D-pad and left stick move the spatial focus,
    // the right stick scrolls the focused scroll container.
    //
    // Only one pointing device highlights the menu at a time — whichever
    // spoke last. A gamepad action clears the mouse hover and flags the
    // documents (.gamepad-nav, which menu.rcss needs to show the :focus
    // highlight); any mouse activity lifts the flag, and hover takes over
    // again on the next cursor move. The RmlUi focus itself is kept across
    // switches so the gamepad resumes right where it left the menu.
    class MenuInputAdapter final : public controller::AbstractKeyboardCallback,
                                   public controller::AbstractGamepadCallback {
    public:
        MenuInputAdapter(
            Rml::Context *context, std::function<void()> on_escape,
            std::function<void(bool)> on_gamepad_nav);

        // While a sink is installed the adapter mutes the menu (no
        // navigation, no clicks, no escape) and hands it every key / button
        // press and every pad axis motion instead; mouse moves, scrolls and
        // releases keep flowing to RmlUi so the cursor stays alive. The
        // controls page uses this to capture a new binding; nullptr
        // uninstalls.
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

        // one navigation step per key: RmlUi acts on the down event
        void send_key(Rml::Input::KeyIdentifier key);

        // Right stick: scrolls the focused element's scroll container
        // like a mouse wheel would under the cursor — the file explorer
        // when the focus is inside it, the whole menu when the window is
        // too small for the panel. The gamepad focus never leaves a
        // scroll container island on its own, so walking up from the
        // focus lands on the container the user is looking at.
        void scroll_at_focus(double value);

        struct AxisNav {
            int direction = 0;// -1 / 0 / +1 after hysteresis
            std::chrono::steady_clock::time_point next_repeat;
        };

        // Hysteresis (engage past 0.5, release under 0.3) turns the analog
        // deflection into clean steps, and holding the stick auto-repeats
        // like a held keyboard arrow; the window dispatches the stick once
        // per frame, which paces the repeat clock.
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
