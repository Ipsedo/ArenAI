//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_DESKTOP_GUI_MENU_H
#define ARENAI_DESKTOP_GUI_MENU_H

#include <filesystem>
#include <memory>

#include <arenai_controller/callback.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/backend.h>

#include "../controller/control_kind.h"

// The gui/ folder is a hexagon of its own: this header is its only public
// port, and it exposes no RmlUi type — the library stays an implementation
// detail of rml_menu.cpp, exactly like GL stays inside arenai_view.
namespace arenai::desktop::gui {

    // what the player can tune in the menu before launching a game
    struct GameSettings {
        int nb_tanks = 16;
        // side length (meters) of the square area the tanks spawn in
        int spawn_side = 500;
        ControllerKind controller_kind = ControllerKind::Keyboard;
        bool fullscreen = false;
        // folder holding the trained SAC state dicts, empty until chosen
        std::filesystem::path sac_folder;
    };

    enum class MenuOutcome { Play, Quit };

    enum class PauseAction { None, Continue, MainMenu, ExitGame };

    // Lives for the whole session (the UI stack is initialised once): the main
    // menu runs its own blocking loop, while the pause popup is driven
    // frame-by-frame by the in-game loop.
    class AbstractGui {
    public:
        virtual ~AbstractGui() = default;

        // polls the window and draws the main menu until Play is clicked
        // (Play) or the window is closed (Quit); presents its own frames
        virtual MenuOutcome run_main_menu() = 0;

        virtual GameSettings settings() const = 0;

        // pause popup, non-blocking: the game loop opens/closes it, renders it
        // over the frozen game frame and polls the clicked button
        virtual void open_pause() = 0;
        virtual void close_pause() = 0;
        // draws the popup over the frame already in the backbuffer; the caller
        // presents afterwards
        virtual void render_pause_overlay() = 0;
        // returns the pending button action and resets it to None
        virtual PauseAction poll_pause_action() = 0;
        // input sink the application routes window events to while paused
        virtual std::shared_ptr<controller::AbstractKeyboardCallback> pause_input() = 0;

        virtual void on_window_resized(int width, int height) = 0;
    };

    std::unique_ptr<AbstractGui> make_gui(
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
        const std::shared_ptr<utils::AbstractResourceFileReader> &asset_reader,
        const GameSettings &initial_settings, int window_width, int window_height);

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_MENU_H
