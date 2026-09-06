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

#include <RmlUi/Core.h>

#include <arenai_view/window.h>

#include "../../../controller/bindings.h"
#include "../../menu.h"
#include "../input.h"

namespace arenai::desktop::gui {

    // one row of the controls page: an action and its current binding
    struct BindingRow {
        Rml::String label;
        Rml::String binding;
        bool listening = false;
        bool bound = true;
    };

    // The Controls screen: the input-kind toggle, the gamepad device list and
    // the keyboard / gamepad binding rows with their capture state machine.
    // Registers its slice of the shared "settings" data model and writes
    // straight into the GameSettings it was built around.
    class ControlsPage {
    public:
        ControlsPage(GameSettings &settings, std::shared_ptr<view::AbstractWindow> window);

        // registers the page's variables and callbacks into the shared model
        void bind(Rml::DataModelConstructor &constructor);
        void set_model_handle(Rml::DataModelHandle handle);
        // captures route the raw input through the menu input adapter, which
        // only exists once the documents are loaded
        void set_input_adapter(std::shared_ptr<MenuInputAdapter> adapter);

        void on_open();
        void on_close();

        // per-tick upkeep of the main-menu loop: gamepad hot-(un)plug while
        // the page is on screen, and the capture timeout
        void update(bool page_visible);

    private:
        std::array<std::optional<KeyboardBinding> *, 5> kb_slots();
        // gamepad slots 1..5 (0 is the fire button)
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
        void assign_gamepad_fire(controller::GamepadButton button);
        void assign_gamepad_axis(GamepadAxis axis, double value);

        void refresh_gamepad_list();

        GameSettings &settings_;
        std::shared_ptr<view::AbstractWindow> window_;
        std::shared_ptr<MenuInputAdapter> input_adapter_;
        Rml::DataModelHandle model_handle_;

        Rml::String controller_display_;
        std::vector<BindingRow> kb_rows_;
        std::vector<BindingRow> gp_rows_;
        std::vector<view::GamepadInfo> gamepads_;
        std::vector<Rml::String> gamepad_names_;
        // index into gamepads_ of the pad feeding the game, -1 when none
        // is connected
        int selected_gamepad_ = -1;
        Rml::String bind_status_;
        bool bind_warning_ = false;
        // slot being captured (-1 = idle) and which page it belongs to
        int capture_slot_ = -1;
        bool capture_keyboard_page_ = false;
        std::chrono::steady_clock::time_point capture_deadline_;
        std::array<bool, NB_GAMEPAD_AXES> axis_armed_{};
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_PAGES_CONTROLS_PAGE_H
