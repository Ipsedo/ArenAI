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

namespace arenai::desktop::gui {

    enum class MenuOutcome { Play, Quit };

    class AbstractMainMenu {
    public:
        virtual ~AbstractMainMenu() = default;

        virtual MenuOutcome run() = 0;

        virtual const GameSettings &settings() const = 0;
    };

    struct MatchStatus {
        float elapsed_seconds = 0.f;
        int player_score = 0;
        // in [0, 1]
        float player_health = 1.f;
        int enemies_max_score = 0;
        int enemies_alive = 0;
        int enemies_total = 0;
    };

    enum class PopupKind { Pause, GameOver, Victory };
    enum class PopupAction { Continue, MainMenu, ExitGame, Retry };
    using PopupListener = std::function<void(PopupAction)>;

    class AbstractPopupMenu {
    public:
        virtual ~AbstractPopupMenu() = default;

        virtual void open(PopupKind kind, const MatchStatus &status, PopupListener on_action) = 0;
        virtual void close() = 0;
        virtual void render() = 0;

        virtual std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_input() = 0;
        virtual std::shared_ptr<controller::AbstractGamepadCallback> gamepad_input() = 0;
    };

    class AbstractLoadingScreen {
    public:
        virtual ~AbstractLoadingScreen() = default;

        virtual void show() = 0;
        virtual void render_frame() = 0;
        virtual void hide() = 0;
    };

    enum class HitKind { Hit, Kill };

    class AbstractHudOverlay {
    public:
        virtual ~AbstractHudOverlay() = default;

        virtual void set_status(const MatchStatus &status) = 0;
        virtual void notify_hit(HitKind kind) = 0;
        virtual void notify_damage(float screen_angle) = 0;
        virtual void set_aim_point(std::optional<glm::vec2> normalized) = 0;
        virtual void render() = 0;
    };

    class AbstractGui {
    public:
        virtual ~AbstractGui() = default;

        virtual AbstractMainMenu &main_menu() = 0;
        virtual AbstractLoadingScreen &loading() = 0;
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
        bool cuda_available;
    };

    std::unique_ptr<AbstractGui> make_gui(
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader, GuiOptions options);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_GUI_H
