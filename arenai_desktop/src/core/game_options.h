#ifndef ARENAI_DESKTOP_CORE_GAME_OPTIONS_H
#define ARENAI_DESKTOP_CORE_GAME_OPTIONS_H

#include <filesystem>

namespace arenai::desktop {

    struct GameOptions {
        int window_width;
        int window_height;
        std::filesystem::path resources_folder;
    };

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_GAME_OPTIONS_H
