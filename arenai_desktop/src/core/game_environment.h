//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_ENVIRONMENT_H
#define ARENAI_DESKTOP_GAME_ENVIRONMENT_H

#include <arenai_core/environment.h>
#include <arenai_model/tank.h>
#include <arenai_view/backend.h>
#include <arenai_view/renderer.h>

#include "../controller/control_kind.h"
#include "../controller/gamepad.h"
#include "../controller/mouse_keyboard.h"

namespace arenai::desktop {

    class DesktopGameEnvironment : public core::BaseTanksEnvironment {
    public:
        DesktopGameEnvironment(
            const std::filesystem::path &asset_folder_path,
            const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
            int nb_tanks, int vision_height, int vision_width, float wanted_frequency,
            const ControllerKind &controller_kind);

        ~DesktopGameEnvironment() override;

        // The environment builds its controller handlers (they need the player
        // tank and the renderer) but does NOT install them on the window:
        // routing the inputs (game vs pause menu) is the application's policy.
        // One of the two is null, depending on the controller kind.
        std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_handler() const;
        std::shared_ptr<controller::AbstractGamepadCallback> gamepad_handler() const;

        // re-renders the player view with the matrices of the last on_draw():
        // used while the game is paused, to draw the frozen scene under the
        // pause menu overlay
        void redraw() const;

        void resize(int width, int height) const;

    protected:
        void
        on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        void on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

        void
        on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

    private:
        std::shared_ptr<view::AbstractWindowedGraphicBackend> windowed_backend;

        std::shared_ptr<utils::AbstractResourceFileReader> asset_file_reader;
        std::unique_ptr<model::PlayerTank> player_tank;
        std::unique_ptr<view::AbstractPlayerRenderer> player_renderer;

        std::shared_ptr<controller::AbstractKeyboardCallback> keyboard_handler_;
        std::shared_ptr<controller::AbstractGamepadCallback> gamepad_handler_;

        std::vector<std::tuple<std::string, glm::mat4>> last_model_matrices_;

        float wanted_frequency;

        ControllerKind controller_kind;
    };

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_GAME_ENVIRONMENT_H
