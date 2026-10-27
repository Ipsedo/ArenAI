//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_DESKTOP_GUI_GUI_H
#define ARENAI_DESKTOP_GUI_GUI_H

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <arenai_controller/callback.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/backend.h>

#include "../core/agent_validator.h"
#include "../core/settings.h"

// The gui/ folder is a hexagon of its own: this header is its only public
// port, and it exposes no RmlUi type — the library stays an implementation
// detail of the rml/ subfolder, exactly like Vulkan stays inside arenai_view.
// The port is split by role so each consumer depends on the slice it uses:
// the application loop on the main menu, the game session on the popups
// and the HUD.
namespace arenai::desktop::gui {

    enum class MenuOutcome { Play, Quit };

    // the main menu screen and its sub-pages (parameters, graphics,
    // controls, AI), driven as a blocking loop that owns the window input
    class AbstractMainMenu {
    public:
        virtual ~AbstractMainMenu() = default;

        virtual MenuOutcome run() = 0;

        // what the player tuned, valid after run() returned
        virtual const GameSettings &settings() const = 0;
    };

    // the match figures the HUD displays every frame and the popups show
    // when they open
    struct MatchStatus {
        float elapsed_seconds = 0.f;
        int player_score = 0;
        // in [0, 1]
        float player_health = 1.f;
        // the enemies' total score over their number (dead ones included)
        float enemies_mean_score = 0.f;
        int enemies_alive = 0;
        int enemies_total = 0;
    };

    enum class PopupKind { Pause, GameOver };
    enum class PopupAction { Continue, MainMenu, ExitGame, Retry };
    using PopupListener = std::function<void(PopupAction)>;

    // the in-game popups drawn over the frozen frame; the game loop keeps
    // polling the window and renders, the popup reports the player's choice
    // through the listener given at open time
    class AbstractPopupMenu {
    public:
        virtual ~AbstractPopupMenu() = default;

        virtual void open(PopupKind kind, const MatchStatus &status, PopupListener on_action) = 0;
        virtual void close() = 0;
        virtual void render() = 0;

        // the sinks to route the window input to while a popup is open
        virtual std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_input() = 0;
        virtual std::shared_ptr<controller::AbstractGamepadCallback> gamepad_input() = 0;
    };

    enum class HitKind { Hit, Kill };

    // the in-game overlay: aim reticle, hit marker, incoming-damage arcs and
    // the match status (clock, both sides' scores, hull, enemies left)
    class AbstractHudOverlay {
    public:
        virtual ~AbstractHudOverlay() = default;

        virtual void set_status(const MatchStatus &status) = 0;
        virtual void notify_hit(HitKind kind) = 0;
        virtual void notify_damage(float screen_angle) = 0;
        // normalized window coordinates; nullopt hides the reticle
        virtual void set_aim_point(std::optional<glm::vec2> normalized) = 0;
        virtual void render() = 0;
    };

    // the whole UI, as the composition root holds it
    class AbstractGui {
    public:
        virtual ~AbstractGui() = default;

        virtual AbstractMainMenu &main_menu() = 0;
        virtual AbstractPopupMenu &popups() = 0;
        virtual AbstractHudOverlay &hud() = 0;

        virtual void on_window_resized(int width, int height) = 0;
    };

    struct GuiOptions {
        GameSettings initial_settings;
        std::vector<AgentPreset> presets;
        std::vector<std::string> gpus;
        int window_width;
        int window_height;
        AgentValidator agent_validator;
        // whether the machine has a CUDA device: off, the AI page's CUDA
        // toggle is disabled and the setting forced to false
        bool cuda_available;
    };

    std::unique_ptr<AbstractGui> make_gui(
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader, GuiOptions options);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_GUI_H
