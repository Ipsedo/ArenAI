//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_ENVIRONMENT_H
#define ARENAI_DESKTOP_GAME_ENVIRONMENT_H

#include <optional>
#include <vector>

#include <arenai_core/environment.h>
#include <arenai_model/tank.h>
#include <arenai_view/backend.h>
#include <arenai_view/renderer.h>

#include "../controller/gamepad.h"
#include "../controller/mouse_keyboard.h"
#include "./settings.h"

namespace arenai::desktop {

    class DesktopGameEnvironment : public core::BaseTanksEnvironment {
    public:
        DesktopGameEnvironment(
            const std::filesystem::path &asset_folder_path,
            const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
            const GameSettings &settings, int vision_height, int vision_width,
            float wanted_frequency);

        ~DesktopGameEnvironment() override;

        // the window-backend half of the reset — the player renderer, its
        // drawable uploads and the input handlers. reset() leaves it out so
        // it can run on a worker thread while the loading screen presents;
        // this must then be called on the main thread, before any frame.
        void create_player_renderer();

        std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_handler() const;
        std::shared_ptr<controller::AbstractGamepadCallback> gamepad_handler() const;

        // one physics frame, the player's view drawn every time; the agents' visions are
        // rendered and returned only when observe is set (the frame before a decision)
        std::optional<std::vector<core::State>>
        play_frame(float time_delta, const std::vector<core::Action> &actions, bool observe);

        void redraw() const;

        void resize(int width, int height) const;

        bool is_player_dead() const;

        int get_score() const;
        // the weakest of the player's life items, in [0, 1] (is_player_dead
        // triggers as soon as one of them dies)
        float player_health_ratio() const;

        // the enemy roster: all of them, the ones still fighting, and the sum
        // of their scores (kills * 10 + hits, as the player's). Non-const:
        // they go through the base class's enemy visitor.
        int nb_enemies();
        int nb_enemies_alive();
        int enemies_total_score();

        model::PlayerHits consume_player_hits() const;
        std::vector<float> consume_damage_screen_angles() const;

        std::optional<glm::vec2> aim_point_on_screen() const;

    protected:
        void
        on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        void on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

        void
        on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

    private:
        std::shared_ptr<view::AbstractWindowedGraphicBackend> windowed_backend;

        // kept from on_reset_drawables for create_player_renderer; owned by
        // the base class, which outlives it
        model::AbstractPhysicEngine *engine_ = nullptr;

        std::shared_ptr<utils::AbstractResourceFileReader> asset_file_reader;
        std::unique_ptr<model::PlayerTank> player_tank;
        std::unique_ptr<view::AbstractPlayerRenderer> player_renderer;

        std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_handler_;
        std::shared_ptr<controller::AbstractGamepadCallback> gamepad_handler_;

        std::vector<std::tuple<std::string, glm::mat4>> last_model_matrices_;

        float wanted_frequency;

        GameSettings settings;
    };

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_GAME_ENVIRONMENT_H
