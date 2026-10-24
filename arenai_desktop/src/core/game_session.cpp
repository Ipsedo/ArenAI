//
// Created by samuel on 25/10/2026.
//

#include "./game_session.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>

#include "../controller/gamepad_selection.h"

namespace arenai::desktop {

    GameSession::GameSession(
        const GameOptions &options, const GameSettings &settings,
        std::shared_ptr<view::AbstractWindowedGraphicBackend> backend, gui::AbstractGui &gui)
        : options_(options), settings_(settings), backend_(std::move(backend)),
          window_(backend_->get_window()), gui_(gui),
          // the menu only lets Play through once the dry-run check passed, so
          // the load succeeds here; a direct launch (ARENAI_DEBUG_AUTOPLAY)
          // surfaces its failure as an exception
          enemy_(load_agent(agent_selection(settings_), options_.cuda)) {
        env_ = std::make_shared<DesktopGameEnvironment>(
            options_.resources_folder, backend_, settings_, enemy_.vision_height,
            enemy_.vision_width, enemy_.wanted_frequency);
        states_ = env_->reset(
            static_cast<float>(settings_.spawn_side), static_cast<float>(settings_.spawn_side));
    }

    void GameSession::install_input() {
        // also covers runs that skip the menu (ARENAI_DEBUG_AUTOPLAY)
        select_configured_gamepad(*window_, settings_);

        router_ = std::make_shared<GameInputRouter>(
            env_->keyboard_handler(), env_->gamepad_handler(), gui_.popups().keyboard_input(),
            gui_.popups().gamepad_input(), [this] { pause_toggle_requested_ = true; },
            settings_.bindings.keyboard);
        window_->set_keyboard_callback(router_);
        window_->set_gamepad_callback(router_);
        apply_cursor_mode();

        window_->set_resize_callback([this](const int width, const int height) {
            gui_.on_window_resized(width, height);
            env_->resize(width, height);
        });
    }

    void GameSession::release_input() {
        gui_.popups().close();
        window_->set_keyboard_callback(nullptr);
        window_->set_gamepad_callback(nullptr);
        window_->set_resize_callback([&gui = gui_](const int width, const int height) {
            gui.on_window_resized(width, height);
        });
    }

    InGameOutcome GameSession::run() {
        install_input();

        while (!window_->should_close() && !outcome_) {
            window_->poll_events();

            if (std::exchange(pause_toggle_requested_, false) && phase_ != Phase::GameOver)
                set_phase(phase_ == Phase::Playing ? Phase::Paused : Phase::Playing);

            if (phase_ == Phase::Playing) play_frame();
            else render_frozen();
        }

        release_input();
        return outcome_.value_or(InGameOutcome::ExitGame);
    }

    void GameSession::set_phase(const Phase phase) {
        phase_ = phase;
        router_->set_paused(phase != Phase::Playing);

        switch (phase) {
            case Phase::Playing: gui_.popups().close(); break;
            case Phase::Paused:
                gui_.popups().open(
                    gui::PopupKind::Pause, env_->get_score(),
                    [this](const gui::PopupAction action) { on_popup_action(action); });
                break;
            case Phase::GameOver:
                gui_.popups().open(
                    gui::PopupKind::GameOver, env_->get_score(),
                    [this](const gui::PopupAction action) { on_popup_action(action); });
                break;
        }
        apply_cursor_mode();
    }

    void GameSession::apply_cursor_mode() const {
        const bool grab =
            phase_ == Phase::Playing && settings_.controller_kind == ControllerKind::Gamepad;
        window_->set_cursor_mode(
            grab ? controller::CursorMode::Disabled : controller::CursorMode::Normal);
    }

    void GameSession::on_popup_action(const gui::PopupAction action) {
        switch (action) {
            case gui::PopupAction::Continue: set_phase(Phase::Playing); break;
            case gui::PopupAction::Retry: outcome_ = InGameOutcome::Retry; break;
            case gui::PopupAction::MainMenu: outcome_ = InGameOutcome::MainMenu; break;
            case gui::PopupAction::ExitGame: outcome_ = InGameOutcome::ExitGame; break;
        }
    }

    void GameSession::render_frozen() const {
        env_->redraw();
        gui_.popups().render();
        backend_->present();
    }

    void GameSession::play_frame() {
        const auto frame_start = std::chrono::steady_clock::now();
        const auto frame_dt =
            std::chrono::milliseconds(static_cast<int>(enemy_.wanted_frequency * 1000.f));

        // the agents decide every action_repeat frames, as in training: their
        // actions are held in between (fire on the decision frame only) and
        // their visions rendered on the frame before the next decision
        const bool decide = frame_ % enemy_.action_repeat == 0;
        const bool observe = (frame_ + 1) % enemy_.action_repeat == 0;
        frame_++;

        if (decide)
            held_actions_ = enemy_.agent->act(states_, enemy_.vision_height, enemy_.vision_width);

        if (auto next_states = env_->play_frame(
                enemy_.wanted_frequency, decide ? held_actions_ : core::release_fire(held_actions_),
                observe))
            states_ = std::move(*next_states);

        auto &hud = gui_.hud();
        if (const auto [hits, kills] = env_->consume_player_hits(); kills > 0)
            hud.notify_hit(gui::HitKind::Kill);
        else if (hits > 0) hud.notify_hit(gui::HitKind::Hit);
        for (const float angle: env_->consume_damage_screen_angles()) hud.notify_damage(angle);
        hud.set_aim_point(env_->aim_point_on_screen());
        hud.render();

        backend_->present();

        if (env_->is_player_dead()) set_phase(Phase::GameOver);

        const auto elapsed = std::chrono::steady_clock::now() - frame_start;
        std::this_thread::sleep_for(
            std::max(frame_dt - elapsed, std::chrono::steady_clock::duration::zero()));
    }

}// namespace arenai::desktop
