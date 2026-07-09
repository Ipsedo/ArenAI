//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_ENVIRONMENT_H
#define ARENAI_DESKTOP_GAME_ENVIRONMENT_H

#include <arenai_core/environment.h>
#include <arenai_model/tank.h>
#include <arenai_view/backend.h>
#include <arenai_view/renderer.h>

#include "../controller/player_controller_handler.h"

namespace arenai::desktop {

    class DesktopGameEnvironment : public core::BaseTanksEnvironment {
    public:
        DesktopGameEnvironment(
            const std::filesystem::path &asset_folder_path,
            const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
            int nb_tanks, int vision_height, int vision_width, float wanted_frequency);

        ~DesktopGameEnvironment() override;

    protected:
        void
        on_draw(const std::vector<std::tuple<std::string, glm::mat4>> &model_matrices) override;

        void on_reset_physics(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

        void
        on_reset_drawables(const std::unique_ptr<model::AbstractPhysicEngine> &engine) override;

    private:
        std::shared_ptr<view::AbstractWindowedGraphicBackend> windowed_backend;

        std::shared_ptr<utils::AbstractFileReader> asset_file_reader;
        std::unique_ptr<model::PlayerTank> player_tank;
        std::unique_ptr<view::AbstractPlayerRenderer> player_renderer;
        std::shared_ptr<MouseKeyboardPlayerControllerHandler> player_controller_handler;

        float wanted_frequency;
    };

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_GAME_ENVIRONMENT_H
