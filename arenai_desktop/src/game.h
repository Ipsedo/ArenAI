//
// Created by samuel on 18/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_H
#define ARENAI_DESKTOP_GAME_H

#include <filesystem>

#include "./gui/menu.h"

namespace arenai::desktop {

    // everything else about the model (vision size, control frequency,
    // hyper-parameters) comes from the selected training run's config.json
    struct ModelOptions {
        std::filesystem::path state_dict_folder;
        bool cuda;
    };

    struct GameOptions {
        int window_width;
        int window_height;
        std::filesystem::path resources_folder;
    };

    enum class InGameOutcome { MainMenu, ExitGame, Retry };

    InGameOutcome run_game(
        const GameOptions &game_options, const ModelOptions &model_options,
        const gui::GameSettings &settings,
        const std::shared_ptr<view::AbstractWindowedGraphicBackend> &graphics_backend,
        const std::unique_ptr<gui::AbstractGui> &gui);

    void run_gui(const GameOptions &game_options, const ModelOptions &model_options);

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_GAME_H
