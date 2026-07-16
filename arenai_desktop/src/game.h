//
// Created by samuel on 18/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_H
#define ARENAI_DESKTOP_GAME_H
#include <filesystem>
#include <map>
#include <string>

#include "./controller/control_kind.h"

namespace arenai::desktop {

    struct ModelOptions {
        int vision_height;
        int vision_width;
        std::map<std::string, std::string> hyper_parameters;
        std::filesystem::path state_dict_folder;
        bool cuda;
    };

    struct GameOptions {
        float wanted_frequency;
        int nb_tanks;
        int window_width;
        int window_height;
        ControllerKind controller_kind;
        std::filesystem::path resources_folder;
    };

    void game_loop(const GameOptions &game_options, const ModelOptions &model_options);

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_GAME_H
