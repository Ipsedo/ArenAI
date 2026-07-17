//
// Created by samuel on 18/03/2026.
//

#include "./game.h"

#include <cstdlib>
#include <iostream>

#include <torch/torch.h>

#include <arenai_core/constants.h>
#include <arenai_train/factory_set.h>
#include <arenai_train/file_reader.h>
#include <arenai_train/torch_converter.h>
#include <arenai_view/backend.h>

#include "./controller/game_input_router.h"
#include "./core/game_environment.h"
#include "./core/user_preferences.h"
#include "./gui/menu.h"

using namespace arenai;

namespace arenai::desktop {

    namespace {

        enum class InGameOutcome { MainMenu, ExitGame };

        // One game session: steps the environment until the window closes or
        // the pause menu asks to leave. While paused, the simulation and the
        // agent are simply not stepped; the frozen frame is re-rendered with
        // the pause popup composited on top.
        InGameOutcome run_game(
            const GameOptions &game_options, const ModelOptions &model_options,
            const gui::GameSettings &settings,
            const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
            const std::unique_ptr<gui::AbstractGui> &gui, const torch::Device device) {
            const auto window = graphics_backend->get_window();

            const auto sac_agent =
                train::SacAgentFactory(model_options.hyper_parameters)
                    .get_agent(
                        model_options.vision_height, model_options.vision_width,
                        model::ENEMY_PROPRIOCEPTION_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
                        model::ENEMY_NB_DISCRETE_ACTION);
            sac_agent->set_train(false);

            sac_agent->load(settings.sac_folder);
            sac_agent->to(device);

            std::cout << "Parameters : " << sac_agent->count_parameters() << std::endl;

            const auto env = std::make_shared<DesktopGameEnvironment>(
                game_options.resources_folder, graphics_backend, settings.nb_tanks,
                model_options.vision_height, model_options.vision_width,
                game_options.wanted_frequency, settings.controller_kind);

            auto states = env->reset(
                static_cast<float>(settings.spawn_side), static_cast<float>(settings.spawn_side));

            // the router owns the window's input slots for the whole session;
            // Escape / Start flip the pause state through toggle_requested
            bool paused = false;
            bool toggle_requested = false;

            const auto router = std::make_shared<GameInputRouter>(
                env->keyboard_handler(), env->gamepad_handler(), gui->pause_input(),
                [&toggle_requested] { toggle_requested = true; });
            window->set_keyboard_callback(router);
            window->set_gamepad_callback(router);

            // the window has a single resize slot: while in game it feeds both
            // the player renderer and the gui overlay
            window->set_resize_callback([&gui, env](const int width, const int height) {
                gui->on_window_resized(width, height);
                env->resize(width, height);
            });

            const auto set_paused = [&](const bool value) {
                paused = value;
                router->set_paused(value);
                if (value) {
                    gui->open_pause();
                    window->set_cursor_mode(controller::CursorMode::Normal);
                } else {
                    gui->close_pause();
                    // in keyboard mode the game handler re-captures the cursor
                    // on its next event
                }
            };

            auto outcome = InGameOutcome::ExitGame;

            // [ARENAI-DBG] temporary auto-repro
            int dbg_auto_frames = -1;
            if (const char *dbg = std::getenv("ARENAI_DEBUG_AUTOFRAMES"))
                dbg_auto_frames = std::atoi(dbg);
            int dbg_frame_count = 0;

            const auto frame_dt =
                std::chrono::milliseconds(static_cast<int>(game_options.wanted_frequency * 1000.f));

            while (!window->should_close()) {
                if (dbg_auto_frames > 0 && dbg_frame_count++ >= dbg_auto_frames) break;
                window->poll_events();

                if (toggle_requested) {
                    toggle_requested = false;
                    set_paused(!paused);
                }

                if (paused) {
                    // frozen scene + popup; pacing comes from the vsync
                    env->redraw();
                    gui->render_pause_overlay();
                    graphics_backend->present();

                    const auto action = gui->poll_pause_action();
                    if (action == gui::PauseAction::Continue) set_paused(false);
                    else if (action == gui::PauseAction::MainMenu) {
                        outcome = InGameOutcome::MainMenu;
                        break;
                    } else if (action == gui::PauseAction::ExitGame) break;

                    continue;
                }

                auto last_time = std::chrono::steady_clock::now();

                const auto [vision, proprioception] = train::states_to_tensor(
                    states, model_options.vision_height, model_options.vision_width);

                const auto [continuous_action, discrete_action] =
                    sac_agent->act(vision.to(device), proprioception.to(device));
                const auto actions_for_env =
                    train::tensor_to_actions(continuous_action.cpu(), discrete_action.cpu());

                const auto steps = env->step(game_options.wanted_frequency, actions_for_env);

                graphics_backend->present();

                states.clear();

                for (const auto &[state, reward, done, is_truncated]: steps)
                    states.push_back(state);

                auto now = std::chrono::steady_clock::now();
                auto dt = now - last_time;

                std::this_thread::sleep_for(
                    std::max(frame_dt - dt, std::chrono::steady_clock::duration::zero()));
            }

            gui->close_pause();
            window->set_keyboard_callback(nullptr);
            window->set_gamepad_callback(nullptr);
            window->set_resize_callback([&gui](const int width, const int height) {
                gui->on_window_resized(width, height);
            });

            return outcome;
        }

    }// namespace

    void game_loop(const GameOptions &game_options, const ModelOptions &model_options) {
        torch::NoGradGuard no_grad;

        const torch::Device device = model_options.cuda ? torch::kCUDA : torch::kCPU;

        // The view owns the window + GL context; the app only speaks the abstract
        // window/backend interface.
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> graphics_backend =
            view::make_glfw_vulkan_backend(
                game_options.window_width, game_options.window_height, "ArenAI");
        const auto window = graphics_backend->get_window();

        std::cout << "Vulkan : " << graphics_backend->renderer_info() << std::endl;

        const auto asset_reader =
            std::make_shared<train::DesktopAssetFileReader>(game_options.resources_folder);

        // the gui (and the RmlUi stack it owns) lives for the whole session:
        // the main menu and the pause popup are two screens of the same context
        // the menu starts from the settings saved on the previous run, falling
        // back to GameSettings' defaults on a first launch
        const auto initial_settings =
            load_preferences({.sac_folder = model_options.state_dict_folder});
        const auto gui = gui::make_gui(
            graphics_backend, asset_reader, initial_settings, game_options.window_width,
            game_options.window_height);

        window->set_resize_callback(
            [&gui](const int width, const int height) { gui->on_window_resized(width, height); });

        // apply the persisted display preference now that the resize callback
        // is wired: the GUI picks the real size up like any user resize
        if (initial_settings.fullscreen) window->set_fullscreen(true);

        // MainMenu <-> InGame state machine; "Main menu" in the pause popup
        // tears the environment down and loops back here, settings preserved
        // [ARENAI-DBG] temporary auto-repro
        const bool dbg_autoplay = std::getenv("ARENAI_DEBUG_AUTOPLAY") != nullptr;

        while (!window->should_close()) {
            const auto menu_outcome = dbg_autoplay ? gui::MenuOutcome::Play : gui->run_main_menu();

            // persisted on every menu exit (Play or Quit) so the tuned
            // settings survive the session whichever way it ends
            save_preferences(gui->settings());

            if (menu_outcome == gui::MenuOutcome::Quit) break;

            if (run_game(
                    game_options, model_options, gui->settings(), graphics_backend, gui, device)
                    == InGameOutcome::ExitGame
                || dbg_autoplay)
                break;
        }
    }

}// namespace arenai::desktop
