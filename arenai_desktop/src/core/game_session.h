//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_CORE_GAME_SESSION_H
#define ARENAI_DESKTOP_CORE_GAME_SESSION_H

#include <memory>
#include <optional>
#include <vector>

#include <arenai_core/environment.h>
#include <arenai_view/backend.h>

#include "../controller/game_input_router.h"
#include "../gui/gui.h"
#include "./agent_loading_checker.h"
#include "./game_environment.h"
#include "./game_options.h"
#include "./settings.h"

namespace arenai::desktop {

    enum class InGameOutcome { MainMenu, ExitGame, Retry };

    // One game, from the agent load to the player leaving: owns the
    // environment and the input routing, and drives the Playing / Paused /
    // GameOver phases against the gui's popups and HUD.
    class GameSession {
    public:
        // loads the enemy agent (throws when the selection is unusable) and
        // builds the environment
        GameSession(
            const GameOptions &options, const GameSettings &settings,
            std::shared_ptr<view::AbstractWindowedGraphicBackend> backend, gui::AbstractGui &gui);

        InGameOutcome run();

    private:
        enum class Phase { Playing, Paused, GameOver };

        void install_input();
        void release_input();

        void set_phase(Phase phase);
        // the cursor follows the phase: free over a popup, grabbed by a
        // gamepad game
        void apply_cursor_mode() const;
        void on_popup_action(gui::PopupAction action);

        void play_frame();
        void render_frozen() const;
        // the figures the HUD and the popups display
        gui::MatchStatus match_status();

        GameOptions options_;
        GameSettings settings_;
        std::shared_ptr<view::AbstractWindowedGraphicBackend> backend_;
        std::shared_ptr<view::AbstractWindow> window_;
        gui::AbstractGui &gui_;

        LoadedAgent enemy_;
        std::shared_ptr<DesktopGameEnvironment> env_;
        std::shared_ptr<GameInputRouter> router_;

        Phase phase_ = Phase::Playing;
        bool pause_toggle_requested_ = false;
        std::optional<InGameOutcome> outcome_;

        int frame_ = 0;
        std::vector<core::State> states_;
        std::vector<core::Action> held_actions_;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_GAME_SESSION_H
