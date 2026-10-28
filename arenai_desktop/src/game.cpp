//
// Created by samuel on 18/03/2026.
//

#include "./game.h"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>

#include <arenai_agent/factory.h>
#include <arenai_agent/file_reader.h>
#include <arenai_view/backend.h>

#include "./core/agent_loading_checker.h"
#include "./core/agent_presets.h"
#include "./core/game_session.h"
#include "./core/user_preferences.h"
#include "./gui/gui.h"

namespace arenai::desktop {

    namespace {

        // Builds a session behind the loading screen: the heavy work (agent,
        // physics, vision threads) runs on a worker while the main thread
        // keeps presenting animation frames, then the window-backend half
        // (finish_loading) runs here, the one place it may touch the
        // swapchain. Returns nullptr when the window closed meanwhile.
        std::unique_ptr<GameSession> load_session(
            const GameOptions &options, const GameSettings &settings,
            const std::shared_ptr<view::AbstractWindowedGraphicBackend> &backend,
            gui::AbstractGui &gui) {
            auto &loading = gui.loading();
            const auto window = backend->get_window();

            loading.show();
            loading.render_frame();

            auto future = std::async(std::launch::async, [&] {
                return std::make_unique<GameSession>(options, settings, backend, gui);
            });

            try {
                while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready
                       && !window->should_close())
                    loading.render_frame();

                // joins the worker in every case; rethrows its exception, the
                // same surfacing a synchronous construction had
                auto session = future.get();

                if (window->should_close()) {
                    loading.hide();
                    return nullptr;
                }

                session->finish_loading();
                loading.hide();
                return session;
            } catch (...) {
                loading.hide();
                throw;
            }
        }

    }// namespace

    void run_gui(const GameOptions &game_options) {
        // the first shipped preset is the out-of-the-box selection
        const auto presets = agent_presets(game_options.resources_folder);
        GameSettings defaults;
        if (!presets.empty()) apply_preset(presets.front(), defaults);

        // loaded before the backend: the window GPU choice only applies at
        // device creation, i.e. here
        const auto initial_settings = load_preferences(defaults, presets);

        const std::shared_ptr graphics_backend = view::make_glfw_vulkan_backend(
            game_options.window_width, game_options.window_height, "ArenAI",
            initial_settings.window_gpu);
        const auto window = graphics_backend->get_window();

        std::cout << "Vulkan : " << graphics_backend->renderer_info() << std::endl;

        const auto asset_reader =
            std::make_shared<agent::DesktopAssetFileReader>(game_options.resources_folder);

        const auto gui = gui::make_gui(
            graphics_backend, asset_reader,
            {.initial_settings = initial_settings,
             .presets = presets,
             .gpus = view::list_vulkan_gpus(),
             .window_width = game_options.window_width,
             .window_height = game_options.window_height,
             .agent_validator =
                 [](const AgentSelection &selection) { return check_agent(selection); },
             .cuda_available = agent::cuda_available()});

        window->set_resize_callback(
            [&gui](const int width, const int height) { gui->on_window_resized(width, height); });

        if (initial_settings.fullscreen) window->set_fullscreen(true);

        const bool dbg_autoplay = std::getenv("ARENAI_DEBUG_AUTOPLAY") != nullptr;

        while (!window->should_close()) {
            const auto menu_outcome =
                dbg_autoplay ? gui::MenuOutcome::Play : gui->main_menu().run();

            save_preferences(gui->main_menu().settings());

            if (menu_outcome == gui::MenuOutcome::Quit) break;

            auto game_outcome = InGameOutcome::ExitGame;
            do {
                const auto session =
                    load_session(game_options, gui->main_menu().settings(), graphics_backend, *gui);
                if (!session) break;
                game_outcome = session->run();
            } while (game_outcome == InGameOutcome::Retry);

            if (game_outcome == InGameOutcome::ExitGame || dbg_autoplay) break;
        }
    }

}// namespace arenai::desktop
