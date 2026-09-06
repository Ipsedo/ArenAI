//
// Created by samuel on 06/09/2026.
//

#include "./controls_page.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <variant>

#include <arenai_controller/input_names.h>

namespace arenai::desktop::gui {

    namespace {

        // slot order of the two pages; the gamepad page puts its button slot
        // (fire) first, then the five axis slots
        constexpr const char *KB_SLOT_LABELS[] = {
            "FORWARD", "BACKWARD", "TURN LEFT", "TURN RIGHT", "FIRE"};
        constexpr const char *GP_SLOT_LABELS[] = {"FIRE",  "STEER",      "AIM X",
                                                  "AIM Y", "ACCELERATE", "REVERSE"};
        constexpr int NB_KB_SLOTS = 5;
        constexpr int NB_GP_SLOTS = 6;
        // accelerate / reverse read a single direction of their axis
        constexpr bool gp_slot_is_one_way(const int slot) { return slot >= 4; }

        // an axis must come back to rest before it can be captured: without
        // this the stick still deflected from navigating the menu (or the A
        // press bound to a trigger) would bind itself instantly
        constexpr double CAPTURE_ENGAGE = 0.6, CAPTURE_REST = 0.3;

        // a capture nobody feeds cancels itself: Escape is the only cancel
        // input (every pad button stays bindable), so a pad-only player
        // needs the timeout
        constexpr auto CAPTURE_TIMEOUT = std::chrono::seconds(15);

        constexpr const char *GAMEPAD_AXIS_LABELS[] = {"L-STICK X", "L-STICK Y", "R-STICK X",
                                                       "R-STICK Y", "LT",        "RT"};

    }// namespace

    ControlsPage::ControlsPage(GameSettings &settings, std::shared_ptr<view::AbstractWindow> window)
        : settings_(settings), window_(std::move(window)),
          controller_display_(
              settings.controller_kind == ControllerKind::Gamepad ? "gamepad" : "keyboard") {}

    void ControlsPage::bind(Rml::DataModelConstructor &constructor) {
        if (auto row_handle = constructor.RegisterStruct<BindingRow>()) {
            row_handle.RegisterMember("label", &BindingRow::label);
            row_handle.RegisterMember("binding", &BindingRow::binding);
            row_handle.RegisterMember("listening", &BindingRow::listening);
            row_handle.RegisterMember("bound", &BindingRow::bound);
        }
        constructor.RegisterArray<std::vector<BindingRow>>();

        constructor.Bind("kb_rows", &kb_rows_);
        constructor.Bind("gp_rows", &gp_rows_);
        constructor.Bind("gamepads", &gamepad_names_);
        constructor.Bind("selected_gamepad", &selected_gamepad_);
        constructor.Bind("bind_status", &bind_status_);
        constructor.Bind("bind_warning", &bind_warning_);
        constructor.Bind("controller", &controller_display_);

        constructor.BindEventCallback(
            "set_controller",
            [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                if (arguments.empty()) return;
                controller_display_ = arguments[0].Get<Rml::String>();
                settings_.controller_kind = controller_display_ == "gamepad"
                                                ? ControllerKind::Gamepad
                                                : ControllerKind::Keyboard;
                model_handle_.DirtyVariable("controller");
                // the toggle lives on the controls page: switching the
                // kind swaps the visible bindings, so a running capture
                // and the status text follow along
                end_capture();
                set_default_bind_status();
                rebuild_binding_rows();
            });
        constructor.BindEventCallback(
            "capture_kb",
            [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                if (!arguments.empty()) begin_capture(true, arguments[0].Get<int>());
            });
        constructor.BindEventCallback(
            "capture_gp",
            [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                if (!arguments.empty()) begin_capture(false, arguments[0].Get<int>());
            });
        constructor.BindEventCallback(
            "select_pad",
            [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &arguments) {
                if (arguments.empty()) return;
                const auto index = static_cast<size_t>(arguments[0].Get<int>());
                if (index >= gamepads_.size()) return;

                settings_.bindings.gamepad.device_guid = gamepads_[index].guid;
                settings_.bindings.gamepad.device_name = gamepads_[index].name;
                window_->select_gamepad(gamepads_[index].id);
                selected_gamepad_ = static_cast<int>(index);
                model_handle_.DirtyVariable("selected_gamepad");
            });
        constructor.BindEventCallback(
            "reset_bindings", [this](Rml::DataModelHandle, Rml::Event &, const Rml::VariantList &) {
                end_capture();
                if (settings_.controller_kind == ControllerKind::Keyboard)
                    settings_.bindings.keyboard = {};
                else {
                    // the device choice is not a binding: keep it
                    auto gamepad = GamepadBindings{};
                    gamepad.device_guid = std::move(settings_.bindings.gamepad.device_guid);
                    gamepad.device_name = std::move(settings_.bindings.gamepad.device_name);
                    settings_.bindings.gamepad = std::move(gamepad);
                }
                set_default_bind_status();
                rebuild_binding_rows();
            });
    }

    void ControlsPage::set_model_handle(const Rml::DataModelHandle handle) {
        model_handle_ = handle;
    }

    void ControlsPage::set_input_adapter(std::shared_ptr<MenuInputAdapter> adapter) {
        input_adapter_ = std::move(adapter);
    }

    void ControlsPage::on_open() {
        refresh_gamepad_list();
        set_default_bind_status();
        rebuild_binding_rows();
    }

    void ControlsPage::on_close() {
        end_capture();
        rebuild_binding_rows();
    }

    void ControlsPage::update(const bool page_visible) {
        // pads can be (un)plugged while the device list is on screen; the
        // refresh is a no-op while nothing changed
        if (page_visible && settings_.controller_kind == ControllerKind::Gamepad)
            refresh_gamepad_list();

        // a forgotten capture ends on its own (a pad-only player has no
        // Escape at hand)
        if (capture_slot_ >= 0 && std::chrono::steady_clock::now() >= capture_deadline_) {
            end_capture();
            rebuild_binding_rows();
        }
    }

    std::array<std::optional<KeyboardBinding> *, 5> ControlsPage::kb_slots() {
        auto &keyboard = settings_.bindings.keyboard;
        return {
            &keyboard.forward, &keyboard.backward, &keyboard.turn_left, &keyboard.turn_right,
            &keyboard.fire};
    }

    std::array<std::optional<GamepadAxisBinding> *, 5> ControlsPage::gp_axis_slots() {
        auto &gamepad = settings_.bindings.gamepad;
        return {
            &gamepad.steer, &gamepad.aim_x, &gamepad.aim_y, &gamepad.accelerate, &gamepad.reverse};
    }

    Rml::String
    ControlsPage::keyboard_slot_label(const std::optional<KeyboardBinding> &slot) const {
        if (!slot) return "UNBOUND";
        if (const auto *button = std::get_if<controller::MouseButton>(&*slot)) switch (*button) {
                case controller::MouseButton::Left: return "MOUSE L";
                case controller::MouseButton::Right: return "MOUSE R";
                case controller::MouseButton::Middle: return "MOUSE M";
            }
        // layout-aware label (Key::Q reads "A" on AZERTY)
        const auto label = window_->key_label(std::get<controller::Key>(*slot));
        return label.empty() ? "?" : Rml::String(label);
    }

    Rml::String ControlsPage::axis_slot_label(
        const std::optional<GamepadAxisBinding> &slot, const bool one_way) {
        if (!slot) return "UNBOUND";
        Rml::String label = GAMEPAD_AXIS_LABELS[static_cast<int>(slot->axis)];
        // a one-way action bound to a stick reads a single direction
        const bool on_stick =
            slot->axis != GamepadAxis::LeftTrigger && slot->axis != GamepadAxis::RightTrigger;
        if (one_way && on_stick) label += slot->sign > 0.f ? "+" : "-";
        return label;
    }

    void ControlsPage::rebuild_binding_rows() {
        kb_rows_.resize(NB_KB_SLOTS);
        const auto keyboard_slots = kb_slots();
        for (int i = 0; i < NB_KB_SLOTS; i++) {
            const bool listening = capture_keyboard_page_ && capture_slot_ == i;
            kb_rows_[i] = {
                .label = KB_SLOT_LABELS[i],
                .binding = listening ? "PRESS A KEY..." : keyboard_slot_label(*keyboard_slots[i]),
                .listening = listening,
                .bound = keyboard_slots[i]->has_value()};
        }

        gp_rows_.resize(NB_GP_SLOTS);
        const auto axis_slots = gp_axis_slots();
        for (int i = 0; i < NB_GP_SLOTS; i++) {
            const bool listening = !capture_keyboard_page_ && capture_slot_ == i;
            const bool fire_slot = i == 0;
            Rml::String binding;
            if (listening) binding = fire_slot ? "PRESS A BUTTON..." : "MOVE AN AXIS...";
            else if (fire_slot)
                binding = settings_.bindings.gamepad.fire
                              ? controller::to_string(*settings_.bindings.gamepad.fire)
                              : "UNBOUND";
            else binding = axis_slot_label(*axis_slots[i - 1], gp_slot_is_one_way(i));
            gp_rows_[i] = {
                .label = GP_SLOT_LABELS[i],
                .binding = std::move(binding),
                .listening = listening,
                .bound = fire_slot ? settings_.bindings.gamepad.fire.has_value()
                                   : axis_slots[i - 1]->has_value()};
        }

        if (model_handle_) {
            model_handle_.DirtyVariable("kb_rows");
            model_handle_.DirtyVariable("gp_rows");
            model_handle_.DirtyVariable("bind_status");
            model_handle_.DirtyVariable("bind_warning");
        }
    }

    void ControlsPage::set_default_bind_status() {
        bind_warning_ = false;
        bind_status_ = settings_.controller_kind == ControllerKind::Keyboard
                           ? "Click a slot, then press the new key or mouse button. "
                             "Escape cancels."
                           : "Click a slot, then press a button or move an axis. "
                             "Escape or 15s of inactivity cancels — Start stays the "
                             "pause toggle.";
    }

    void ControlsPage::begin_capture(const bool keyboard_page, const int slot) {
        const int nb_slots = keyboard_page ? NB_KB_SLOTS : NB_GP_SLOTS;
        if (slot < 0 || slot >= nb_slots) return;

        capture_keyboard_page_ = keyboard_page;
        capture_slot_ = slot;
        capture_deadline_ = std::chrono::steady_clock::now() + CAPTURE_TIMEOUT;
        // every axis must return to rest once before it can bind (the
        // stick may still be deflected from navigating the menu)
        axis_armed_.fill(false);
        set_default_bind_status();

        input_adapter_->set_capture_sink(
            [this](const RawMenuInput &input) { on_capture_input(input); });
        rebuild_binding_rows();
    }

    void ControlsPage::end_capture() {
        capture_slot_ = -1;
        if (input_adapter_) input_adapter_->set_capture_sink(nullptr);
    }

    void ControlsPage::on_capture_input(const RawMenuInput &input) {
        // Escape is the only cancel input (every pad button stays
        // bindable); a pad-only player relies on the capture timeout
        if (const auto *key = std::get_if<controller::Key>(&input);
            key != nullptr && *key == controller::Key::Escape) {
            end_capture();
            rebuild_binding_rows();
            return;
        }

        if (capture_keyboard_page_) {
            if (const auto *key = std::get_if<controller::Key>(&input))
                assign_keyboard(KeyboardBinding(*key));
            else if (const auto *button = std::get_if<controller::MouseButton>(&input))
                assign_keyboard(KeyboardBinding(*button));
            // pad input has no meaning on the keyboard page
            return;
        }

        if (capture_slot_ == 0) {
            // Start stays the in-game pause toggle, never a binding
            if (const auto *button = std::get_if<controller::GamepadButton>(&input);
                button != nullptr && *button != controller::GamepadButton::Start)
                assign_gamepad_fire(*button);
            return;
        }

        if (const auto *motion = std::get_if<std::pair<GamepadAxis, double>>(&input)) {
            const auto &[axis, value] = *motion;
            auto &armed = axis_armed_[static_cast<size_t>(axis)];
            if (std::abs(value) < CAPTURE_REST) armed = true;
            else if (armed && std::abs(value) > CAPTURE_ENGAGE) assign_gamepad_axis(axis, value);
        }
    }

    void ControlsPage::unbind_conflict(const Rml::String &new_label, const Rml::String &old_label) {
        bind_warning_ = true;
        bind_status_ =
            new_label + " was bound to " + old_label + " — " + old_label + " is now unbound.";
    }

    void ControlsPage::assign_keyboard(const KeyboardBinding &binding) {
        const auto slots = kb_slots();
        *slots[capture_slot_] = binding;

        for (int i = 0; i < NB_KB_SLOTS; i++)
            if (i != capture_slot_ && *slots[i] == binding) {
                *slots[i] = std::nullopt;
                unbind_conflict(keyboard_slot_label(binding), KB_SLOT_LABELS[i]);
            }

        end_capture();
        rebuild_binding_rows();
    }

    void ControlsPage::assign_gamepad_fire(const controller::GamepadButton button) {
        // single button slot: no conflict possible
        settings_.bindings.gamepad.fire = button;
        end_capture();
        rebuild_binding_rows();
    }

    void ControlsPage::assign_gamepad_axis(const GamepadAxis axis, const double value) {
        const bool one_way = gp_slot_is_one_way(capture_slot_);
        const auto slots = gp_axis_slots();
        const GamepadAxisBinding binding{.axis = axis, .sign = one_way && value < 0. ? -1.f : 1.f};
        *slots[capture_slot_ - 1] = binding;

        // two slots clash when they read the same range of an axis: a
        // two-way slot owns the whole axis, one-way slots only their
        // captured side
        for (int slot = 1; slot < NB_GP_SLOTS; slot++) {
            if (slot == capture_slot_) continue;
            auto &other = *slots[slot - 1];
            if (!other || other->axis != axis) continue;
            if (one_way && gp_slot_is_one_way(slot) && other->sign != binding.sign) continue;
            other = std::nullopt;
            unbind_conflict(axis_slot_label(binding, one_way), GP_SLOT_LABELS[slot]);
        }

        end_capture();
        rebuild_binding_rows();
    }

    void ControlsPage::refresh_gamepad_list() {
        auto gamepads = window_->list_gamepads();
        const bool changed = gamepads.size() != gamepads_.size()
                             || !std::equal(
                                 gamepads.begin(), gamepads.end(), gamepads_.begin(),
                                 [](const view::GamepadInfo &a, const view::GamepadInfo &b) {
                                     return a.id == b.id && a.guid == b.guid && a.name == b.name;
                                 });
        if (!changed) return;

        gamepads_ = std::move(gamepads);
        gamepad_names_.clear();
        for (const auto &pad: gamepads_) gamepad_names_.emplace_back(pad.name);

        // the preferred pad when connected, else the first one (which
        // is what the window falls back to)
        selected_gamepad_ = gamepads_.empty() ? -1 : 0;
        for (size_t i = 0; i < gamepads_.size(); i++)
            if (!settings_.bindings.gamepad.device_guid.empty()
                && gamepads_[i].guid == settings_.bindings.gamepad.device_guid) {
                selected_gamepad_ = static_cast<int>(i);
                break;
            }
        window_->select_gamepad(selected_gamepad_ >= 0 ? gamepads_[selected_gamepad_].id : -1);

        if (model_handle_) {
            model_handle_.DirtyVariable("gamepads");
            model_handle_.DirtyVariable("selected_gamepad");
        }
    }

}// namespace arenai::desktop::gui
