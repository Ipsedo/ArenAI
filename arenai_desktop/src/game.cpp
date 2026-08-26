//
// Created by samuel on 18/03/2026.
//

#include "./game.h"

#include <cstdlib>
#include <iostream>

#include <arenai_agent/factory_set.h>
#include <arenai_agent/file_reader.h>
#include <arenai_model/constants.h>
#include <arenai_view/backend.h>

#include "./controller/game_input_router.h"
#include "./core/agent_loading_checker.h"
#include "./core/game_environment.h"
#include "./core/user_preferences.h"
#include "./gui/menu.h"

using namespace arenai;

namespace arenai::desktop {

    InGameOutcome run_game(
        const GameOptions &game_options, const ModelOptions &model_options,
        const gui::GameSettings &settings,
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
        const std::unique_ptr<gui::AbstractGui> &gui) {
        const auto window = graphics_backend->get_window();

        const std::shared_ptr<agent::AbstractAgent> sac_agent =
            agent::ActorAgentFactory(model_options.hyper_parameters)
                .get_agent(
                    model_options.vision_height, model_options.vision_width,
                    model::ENEMY_PROPRIOCEPTION_SIZE, model::ENEMY_NB_CONTINUOUS_ACTION,
                    model::ENEMY_NB_DISCRETE_ACTION);

        sac_agent->load(settings.sac_folder);

        // route the pad input to the configured device when it is connected
        // (also covers runs that skip the menu, e.g. ARENAI_DEBUG_AUTOPLAY)
        if (settings.controller_kind == ControllerKind::Gamepad
            && !settings.bindings.gamepad.device_guid.empty())
            for (const auto &[id, name, guid]: window->list_gamepads())
                if (guid == settings.bindings.gamepad.device_guid) {
                    window->select_gamepad(id);
                    break;
                }

        const auto env = std::make_shared<DesktopGameEnvironment>(
            game_options.resources_folder, graphics_backend, settings.nb_tanks,
            model_options.vision_height, model_options.vision_width, game_options.wanted_frequency,
            settings.controller_kind, settings.bindings);

        auto states = env->reset(
            static_cast<float>(settings.spawn_side), static_cast<float>(settings.spawn_side));

        bool paused = false;
        bool game_over = false;
        bool toggle_requested = false;

        const auto router = std::make_shared<GameInputRouter>(
            env->keyboard_handler(), env->gamepad_handler(), gui->pause_input(),
            gui->pause_gamepad_input(), [&toggle_requested] { toggle_requested = true; },
            settings.bindings.keyboard);
        window->set_keyboard_callback(router);
        window->set_gamepad_callback(router);

        if (settings.controller_kind == ControllerKind::Gamepad)
            window->set_cursor_mode(controller::CursorMode::Disabled);

        window->set_resize_callback([&gui, env](const int width, const int height) {
            gui->on_window_resized(width, height);
            env->resize(width, height);
        });

        const auto set_paused = [&](const bool value) {
            paused = value;
            router->set_paused(value);
            if (value) {
                gui->open_pause(env->get_score());
                window->set_cursor_mode(controller::CursorMode::Normal);
            } else {
                gui->close_pause();

                if (settings.controller_kind == ControllerKind::Gamepad)
                    window->set_cursor_mode(controller::CursorMode::Disabled);
            }
        };

        const auto set_game_over = [&] {
            game_over = true;
            router->set_paused(true);
            gui->open_game_over(env->get_score());
            window->set_cursor_mode(controller::CursorMode::Normal);
        };

        auto outcome = InGameOutcome::ExitGame;

        const auto frame_dt =
            std::chrono::milliseconds(static_cast<int>(game_options.wanted_frequency * 1000.f));

        while (!window->should_close()) {
            window->poll_events();

            if (toggle_requested) {
                toggle_requested = false;
                if (!game_over) set_paused(!paused);
            }

            if (paused || game_over) {
                env->redraw();
                gui->render_pause_overlay();
                graphics_backend->present();

                if (const auto action = gui->poll_pause_action();
                    action == gui::PauseAction::Continue)
                    set_paused(false);
                else if (action == gui::PauseAction::Retry) {
                    outcome = InGameOutcome::Retry;
                    break;
                } else if (action == gui::PauseAction::MainMenu) {
                    outcome = InGameOutcome::MainMenu;
                    break;
                } else if (action == gui::PauseAction::ExitGame) break;

                continue;
            }

            auto last_time = std::chrono::steady_clock::now();

            const auto action =
                sac_agent->act(states, model_options.vision_height, model_options.vision_width);

            const auto steps = env->step(game_options.wanted_frequency, action);

            if (const auto [hits, kills] = env->consume_player_hits(); kills > 0)
                gui->notify_hit(gui::HitKind::Kill);
            else if (hits > 0) gui->notify_hit(gui::HitKind::Hit);
            gui->set_aim_point(env->aim_point_on_screen());
            gui->render_hud_overlay();

            graphics_backend->present();

            if (env->is_player_dead()) set_game_over();

            states.clear();

            for (const auto &[state, reward, done]: steps) states.push_back(state);

            auto now = std::chrono::steady_clock::now();
            auto dt = now - last_time;

            std::this_thread::sleep_for(
                std::max(frame_dt - dt, std::chrono::steady_clock::duration::zero()));
        }

        gui->close_pause();
        gui->close_game_over();
        window->set_keyboard_callback(nullptr);
        window->set_gamepad_callback(nullptr);
        window->set_resize_callback(
            [&gui](const int width, const int height) { gui->on_window_resized(width, height); });

        return outcome;
    }

    void run_gui(const GameOptions &game_options, const ModelOptions &model_options) {
        const std::shared_ptr graphics_backend = view::make_glfw_vulkan_backend(
            game_options.window_width, game_options.window_height, "ArenAI");
        const auto window = graphics_backend->get_window();

        std::cout << "Vulkan : " << graphics_backend->renderer_info() << std::endl;

        const auto asset_reader =
            std::make_shared<agent::DesktopAssetFileReader>(game_options.resources_folder);

        const auto initial_settings =
            load_preferences({.sac_folder = model_options.state_dict_folder});
        const auto gui = gui::make_gui(
            graphics_backend, asset_reader, initial_settings, game_options.window_width,
            game_options.window_height, [&model_options](const std::filesystem::path &folder) {
                return check_agent_folder(model_options, folder);
            });

        window->set_resize_callback(
            [&gui](const int width, const int height) { gui->on_window_resized(width, height); });

        if (initial_settings.fullscreen) window->set_fullscreen(true);

        const bool dbg_autoplay = std::getenv("ARENAI_DEBUG_AUTOPLAY") != nullptr;

        while (!window->should_close()) {
            const auto menu_outcome = dbg_autoplay ? gui::MenuOutcome::Play : gui->run_main_menu();

            save_preferences(gui->settings());

            if (menu_outcome == gui::MenuOutcome::Quit) break;

            InGameOutcome game_outcome;
            do {
                game_outcome =
                    run_game(game_options, model_options, gui->settings(), graphics_backend, gui);
            } while (game_outcome == InGameOutcome::Retry);

            if (game_outcome == InGameOutcome::ExitGame || dbg_autoplay) break;
        }
    }

}// namespace arenai::desktop
