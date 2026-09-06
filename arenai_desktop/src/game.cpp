//
// Created by samuel on 18/03/2026.
//

#include "./game.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include <arenai_agent/factory.h>
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

        // the menu only lets Play through once check_agent() validated the
        // selection, so the config.json is there and the load succeeds
        const gui::AgentSelection selection = {
            .config = settings.agent_config,
            .folder = settings.agent_folder,
            .algorithm = settings.agent_algorithm};
        const auto config_path = resolve_agent_config(selection);
        if (!config_path)
            throw std::runtime_error(
                "config.json not found in " + selection.folder.string() + " or its parent");

        std::ifstream config_stream(*config_path);
        const auto config = nlohmann::json::parse(config_stream);

        agent::AgentFactory factory(config);
        const int vision_height = factory.get_vision_height();
        const int vision_width = factory.get_vision_width();
        const float wanted_frequency = factory.get_wanted_frequency();

        const std::shared_ptr<agent::AbstractAgent> enemy_agent = factory.get_agent(
            to_agent_algorithm(settings.agent_algorithm), model::ENEMY_PROPRIOCEPTION_SIZE,
            model::ENEMY_NB_CONTINUOUS_ACTION, model::ENEMY_NB_DISCRETE_ACTION, model_options.cuda);

        enemy_agent->load(settings.agent_folder);

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
            game_options.resources_folder, graphics_backend, settings, vision_height, vision_width,
            wanted_frequency);

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
            std::chrono::milliseconds(static_cast<int>(wanted_frequency * 1000.f));

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

            const auto action = enemy_agent->act(states, vision_height, vision_width);

            const auto steps = env->step(wanted_frequency, action);

            if (const auto [hits, kills] = env->consume_player_hits(); kills > 0)
                gui->notify_hit(gui::HitKind::Kill);
            else if (hits > 0) gui->notify_hit(gui::HitKind::Hit);

            for (const float angle: env->consume_damage_screen_angles()) gui->notify_damage(angle);

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
        // loaded before the backend: the window GPU choice only applies at
        // device creation, i.e. here
        const auto initial_settings = load_preferences(
            {.agent_folder = model_options.state_dict_folder,
             .agent_config = model_options.config_json});

        const std::shared_ptr graphics_backend = view::make_glfw_vulkan_backend(
            game_options.window_width, game_options.window_height, "ArenAI",
            initial_settings.window_gpu);
        const auto window = graphics_backend->get_window();

        std::cout << "Vulkan : " << graphics_backend->renderer_info() << std::endl;

        const auto asset_reader =
            std::make_shared<agent::DesktopAssetFileReader>(game_options.resources_folder);

        const auto gui = gui::make_gui(
            graphics_backend, asset_reader, initial_settings, view::list_vulkan_gpus(),
            game_options.window_width, game_options.window_height,
            [](const gui::AgentSelection &selection) { return check_agent(selection); });

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
