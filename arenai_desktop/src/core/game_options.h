//
// Created by samuel on 24/10/2026.
//

#ifndef ARENAI_DESKTOP_CORE_GAME_OPTIONS_H
#define ARENAI_DESKTOP_CORE_GAME_OPTIONS_H

#include <filesystem>

namespace arenai::desktop {

    // what belongs to the player's machine (command line); everything about
    // the model (which one, vision size, control frequency, hyper-parameters)
    // comes from the preset or folder selected in the menu and its config.json
    struct GameOptions {
        int window_width;
        int window_height;
        std::filesystem::path resources_folder;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_GAME_OPTIONS_H
