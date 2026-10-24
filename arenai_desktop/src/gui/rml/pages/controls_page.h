//
// Created by samuel on 06/09/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_PAGES_CONTROLS_PAGE_H
#define ARENAI_DESKTOP_GUI_RML_PAGES_CONTROLS_PAGE_H

#include <array>
#include <chrono>
#include <memory>
#include <optional>
#include <vector>

#include <arenai_view/window.h>

#include "../../../controller/bindings.h"
#include "../input.h"
#include "../menu_page.h"
#include "../menu_state.h"

namespace arenai::desktop::gui {

    // one row of the controls page: an action and its current binding
    struct BindingRow {
        Rml::String label;
        Rml::String binding;
        bool listening = false;
        bool bound = true;
    };

    // The Controls screen: input kind, gamepad device, and the binding slots
    // captured through the menu input adapter's raw sink.
    class ControlsPage final : public MenuPage {
    public:
        ControlsPage(
            MenuState &state, std::shared_ptr<view::AbstractWindow> window,
            MenuInputAdapter &input_adapter);

        void update() override;

    protected:
        void bind_controls(DataModel &model) override;
        void on_show() override;
        void on_hide() override;

    private:
        std::array<std::optional<KeyboardBinding> *, 6> kb_slots();
        std::array<std::optional<controller::GamepadButton> *, 2> gp_button_slots();
        std::array<std::optional<GamepadAxisBinding> *, 5> gp_axis_slots();

        Rml::String keyboard_slot_label(const std::optional<KeyboardBinding> &slot) const;
        static Rml::String
        axis_slot_label(const std::optional<GamepadAxisBinding> &slot, bool one_way);

        void rebuild_binding_rows();
        void set_default_bind_status();

        void begin_capture(bool keyboard_page, int slot);
        void end_capture();
        void on_capture_input(const RawMenuInput &input);

        void unbind_conflict(const Rml::String &new_label, const Rml::String &old_label);
        void assign_keyboard(const KeyboardBinding &binding);
        void assign_gamepad_button(controller::GamepadButton button);
        void assign_gamepad_axis(GamepadAxis axis, double value);

        void refresh_gamepad_list();

        GameSettings &settings_;
        std::shared_ptr<view::AbstractWindow> window_;
        MenuInputAdapter &input_adapter_;

        Rml::String controller_display_;
        std::vector<BindingRow> kb_rows_;
        std::vector<BindingRow> gp_rows_;
        std::vector<view::GamepadInfo> gamepads_;
        std::vector<Rml::String> gamepad_names_;

        int selected_gamepad_ = -1;
        Rml::String bind_status_;
        bool bind_warning_ = false;

        int capture_slot_ = -1;
        bool capture_keyboard_page_ = false;
        std::chrono::steady_clock::time_point capture_deadline_;
        std::array<bool, NB_GAMEPAD_AXES> axis_armed_{};
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_CONTROLS_PAGE_H
