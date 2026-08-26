//
// Created by samuel on 17/07/2026.
//

#include "./user_preferences.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <utility>

#include <nlohmann/json.hpp>

#include <arenai_controller/input_names.h>

namespace arenai::desktop {

    namespace {

        // ---- bindings <-> canonical strings -------------------------------
        // Serialized names: keys/buttons/axes by their canonical name, mouse
        // buttons as "MouseLeft/Right/Middle", one-way axes captured on the
        // negative side with a trailing '-'. "" = deliberately unbound;
        // missing or unrecognized falls back to the default slot.

        constexpr std::pair<controller::MouseButton, const char *> MOUSE_BUTTON_NAMES[] = {
            {controller::MouseButton::Left, "MouseLeft"},
            {controller::MouseButton::Right, "MouseRight"},
            {controller::MouseButton::Middle, "MouseMiddle"},
        };

        std::string keyboard_binding_to_string(const std::optional<KeyboardBinding> &slot) {
            if (!slot) return "";
            if (const auto *button = std::get_if<controller::MouseButton>(&*slot))
                for (const auto &[value, name]: MOUSE_BUTTON_NAMES)
                    if (value == *button) return name;
            return controller::to_string(std::get<controller::Key>(*slot));
        }

        void load_keyboard_binding(
            const nlohmann::json &json, const char *field, std::optional<KeyboardBinding> &slot) {
            if (!json.contains(field)) return;
            const auto name = json.value(field, std::string());
            if (name.empty()) {
                slot = std::nullopt;
                return;
            }
            for (const auto &[value, button_name]: MOUSE_BUTTON_NAMES)
                if (name == button_name) {
                    slot = value;
                    return;
                }
            if (const auto key = controller::key_from_string(name); key != controller::Key::Unknown)
                slot = key;
        }

        std::string axis_binding_to_string(const std::optional<GamepadAxisBinding> &slot) {
            if (!slot) return "";
            return std::string(to_string(slot->axis)) + (slot->sign < 0.f ? "-" : "");
        }

        void load_axis_binding(
            const nlohmann::json &json, const char *field,
            std::optional<GamepadAxisBinding> &slot) {
            if (!json.contains(field)) return;
            auto name = json.value(field, std::string());
            if (name.empty()) {
                slot = std::nullopt;
                return;
            }
            float sign = 1.f;
            if (name.back() == '-') {
                sign = -1.f;
                name.pop_back();
            }
            if (const auto axis = gamepad_axis_from_string(name))
                slot = GamepadAxisBinding{.axis = *axis, .sign = sign};
        }

        void load_bindings(const nlohmann::json &json, ControlBindings &bindings) {
            if (const auto keyboard = json.value("keyboard", nlohmann::json::object());
                keyboard.is_object()) {
                load_keyboard_binding(keyboard, "forward", bindings.keyboard.forward);
                load_keyboard_binding(keyboard, "backward", bindings.keyboard.backward);
                load_keyboard_binding(keyboard, "turn_left", bindings.keyboard.turn_left);
                load_keyboard_binding(keyboard, "turn_right", bindings.keyboard.turn_right);
                load_keyboard_binding(keyboard, "fire", bindings.keyboard.fire);
            }

            if (const auto gamepad = json.value("gamepad", nlohmann::json::object());
                gamepad.is_object()) {
                if (gamepad.contains("fire")) {
                    const auto name = gamepad.value("fire", std::string());
                    if (name.empty()) bindings.gamepad.fire = std::nullopt;
                    else if (const auto button = controller::gamepad_button_from_string(name))
                        bindings.gamepad.fire = *button;
                }
                load_axis_binding(gamepad, "steer", bindings.gamepad.steer);
                load_axis_binding(gamepad, "aim_x", bindings.gamepad.aim_x);
                load_axis_binding(gamepad, "aim_y", bindings.gamepad.aim_y);
                load_axis_binding(gamepad, "accelerate", bindings.gamepad.accelerate);
                load_axis_binding(gamepad, "reverse", bindings.gamepad.reverse);
                bindings.gamepad.device_guid =
                    gamepad.value("device_guid", bindings.gamepad.device_guid);
                bindings.gamepad.device_name =
                    gamepad.value("device_name", bindings.gamepad.device_name);
            }
        }

        nlohmann::json bindings_to_json(const ControlBindings &bindings) {
            return {
                {"keyboard",
                 {{"forward", keyboard_binding_to_string(bindings.keyboard.forward)},
                  {"backward", keyboard_binding_to_string(bindings.keyboard.backward)},
                  {"turn_left", keyboard_binding_to_string(bindings.keyboard.turn_left)},
                  {"turn_right", keyboard_binding_to_string(bindings.keyboard.turn_right)},
                  {"fire", keyboard_binding_to_string(bindings.keyboard.fire)}}},
                {"gamepad",
                 {{"fire",
                   bindings.gamepad.fire ? controller::to_string(*bindings.gamepad.fire) : ""},
                  {"steer", axis_binding_to_string(bindings.gamepad.steer)},
                  {"aim_x", axis_binding_to_string(bindings.gamepad.aim_x)},
                  {"aim_y", axis_binding_to_string(bindings.gamepad.aim_y)},
                  {"accelerate", axis_binding_to_string(bindings.gamepad.accelerate)},
                  {"reverse", axis_binding_to_string(bindings.gamepad.reverse)},
                  {"device_guid", bindings.gamepad.device_guid},
                  {"device_name", bindings.gamepad.device_name}}},
            };
        }

        std::filesystem::path user_cache_dir() {
#ifdef _WIN32
            if (const char *local_app_data = std::getenv("LOCALAPPDATA"))
                return std::filesystem::path(local_app_data) / "ArenAI";
#else
            if (const char *xdg_cache = std::getenv("XDG_CACHE_HOME"))
                return std::filesystem::path(xdg_cache) / "arenai";
            if (const char *home = std::getenv("HOME"))
                return std::filesystem::path(home) / ".cache" / "arenai";
#endif
            // no user profile at all (stripped-down service environment)
            return std::filesystem::temp_directory_path() / "arenai";
        }

    }// namespace

    std::filesystem::path preferences_path() { return user_cache_dir() / "preferences.json"; }

    gui::GameSettings load_preferences(const gui::GameSettings &defaults) {
        auto settings = defaults;

        const auto path = preferences_path();

        try {
            std::ifstream file(path);
            if (!file.is_open()) return settings;// first launch, nothing saved yet

            const auto json = nlohmann::json::parse(file);

            if (const int nb_tanks = json.value("nb_tanks", settings.nb_tanks); nb_tanks > 0)
                settings.nb_tanks = nb_tanks;
            if (const int spawn_side = json.value("spawn_side", settings.spawn_side);
                spawn_side > 0)
                settings.spawn_side = spawn_side;

            settings.controller_kind = json.value("controller", std::string()) == "gamepad"
                                           ? ControllerKind::Gamepad
                                           : ControllerKind::Keyboard;

            settings.fullscreen = json.value("fullscreen", settings.fullscreen);

            if (const auto bindings = json.value("bindings", nlohmann::json::object());
                bindings.is_object())
                load_bindings(bindings, settings.bindings);

            // a stale folder (moved, deleted, unplugged drive) falls back to
            // the default so the menu never starts on an unplayable selection
            if (const std::filesystem::path sac_folder = json.value("sac_folder", std::string());
                !sac_folder.empty() && std::filesystem::is_directory(sac_folder))
                settings.sac_folder = sac_folder;
        } catch (const std::exception &e) {
            std::cerr << "Cannot load preferences " << path << ": " << e.what() << std::endl;
            return defaults;
        }

        return settings;
    }

    void save_preferences(const gui::GameSettings &settings) {
        const auto path = preferences_path();

        try {
            const nlohmann::json json = {
                {"nb_tanks", settings.nb_tanks},
                {"spawn_side", settings.spawn_side},
                {"controller",
                 settings.controller_kind == ControllerKind::Gamepad ? "gamepad" : "keyboard"},
                {"fullscreen", settings.fullscreen},
                {"bindings", bindings_to_json(settings.bindings)},
                {"sac_folder", settings.sac_folder.string()},
            };

            std::filesystem::create_directories(path.parent_path());

            std::ofstream file(path);
            if (!file.is_open()) throw std::runtime_error("cannot open the file for writing");
            file << json.dump(4) << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Cannot save preferences " << path << ": " << e.what() << std::endl;
        }
    }

}// namespace arenai::desktop
