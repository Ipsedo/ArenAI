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

        void create_player_renderer();

        std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_handler() const;
        std::shared_ptr<controller::AbstractGamepadCallback> gamepad_handler() const;

        std::optional<std::vector<core::State>>
        play_frame(float time_delta, const std::vector<core::Action> &actions, bool observe);

        void redraw() const;

        void resize(int width, int height) const;

        bool is_player_dead() const;

        int get_score() const;
        float player_health_ratio() const;

        int nb_enemies();
        int nb_enemies_alive();
        int enemies_max_score();

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
