//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_DESKTOP_GUI_MENU_H
#define ARENAI_DESKTOP_GUI_MENU_H

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_controller/callback.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/backend.h>

#include "../controller/bindings.h"
#include "../controller/control_kind.h"

// The gui/ folder is a hexagon of its own: this header is its only public
// port, and it exposes no RmlUi type — the library stays an implementation
// detail of rml_menu.cpp and the rml/ subfolder, exactly like GL stays
// inside arenai_view.
namespace arenai::desktop::gui {

    // shadow quality presets; Ultra matches the historical hardcoded rendering
    enum class ShadowQuality { Off, Low, Medium, High, Ultra };

    // canonical names, shared by the JSON preferences and the menu bindings
    constexpr const char *to_string(const ShadowQuality quality) {
        switch (quality) {
            case ShadowQuality::Off: return "off";
            case ShadowQuality::Low: return "low";
            case ShadowQuality::Medium: return "medium";
            case ShadowQuality::High: return "high";
            default: return "ultra";
        }
    }

    constexpr std::optional<ShadowQuality> shadow_quality_from_string(std::string_view name) {
        for (const auto quality:
             {ShadowQuality::Off, ShadowQuality::Low, ShadowQuality::Medium, ShadowQuality::High,
              ShadowQuality::Ultra})
            if (name == to_string(quality)) return quality;
        return std::nullopt;
    }

    // shadow atlas side per preset (0 = shadows disabled, the size is unused)
    constexpr int shadow_map_size(const ShadowQuality quality) {
        switch (quality) {
            case ShadowQuality::Off: return 0;
            case ShadowQuality::Low: return 2048;
            case ShadowQuality::Medium: return 4096;
            case ShadowQuality::High: return 8192;
            default: return 16384;
        }
    }

    // what the player can tune in the menu before launching a game
    struct GameSettings {
        int nb_tanks = 16;
        int spawn_side = 500;

        ControllerKind controller_kind = ControllerKind::Keyboard;
        ControlBindings bindings;

        bool fullscreen = false;

        ShadowQuality shadow_quality = ShadowQuality::Ultra;
        int msaa_samples = 4;

        std::string window_gpu;
        std::string vision_gpu;

        std::filesystem::path sac_folder;
    };

    enum class MenuOutcome { Play, Quit };

    enum class PauseAction { None, Continue, MainMenu, ExitGame, Retry };

    enum class HitKind { Hit, Kill };

    class AbstractGui {
    public:
        virtual ~AbstractGui() = default;

        virtual MenuOutcome run_main_menu() = 0;

        virtual GameSettings settings() const = 0;

        virtual void open_pause(int score) = 0;
        virtual void close_pause() = 0;
        virtual void render_pause_overlay() = 0;

        virtual void open_game_over(int score) = 0;
        virtual void close_game_over() = 0;

        virtual PauseAction poll_pause_action() = 0;

        virtual void notify_hit(HitKind kind) = 0;
        virtual void notify_damage(float screen_angle) = 0;

        virtual void set_aim_point(std::optional<glm::vec2> normalized) = 0;
        virtual void render_hud_overlay() = 0;

        virtual std::shared_ptr<controller::AbstractKeyboardCallback> pause_input() = 0;
        virtual std::shared_ptr<controller::AbstractGamepadCallback> pause_gamepad_input() = 0;

        virtual void on_window_resized(int width, int height) = 0;
    };

    using SacFolderValidator =
        std::function<std::optional<std::string>(const std::filesystem::path &)>;

    std::unique_ptr<AbstractGui> make_gui(
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
        const GameSettings &initial_settings, const std::vector<std::string> &gpus,
        int window_width, int window_height, SacFolderValidator sac_validator);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_MENU_H
