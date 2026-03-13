//
// Created by samuel on 11/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_ENVIRONMENT_H
#define ARENAI_DESKTOP_GAME_ENVIRONMENT_H

#include <arenai_core/environment.h>
#include <arenai_view/renderer.h>

class DesktopGameEnvironment : public BaseTanksEnvironment {
public:
    DesktopGameEnvironment(
        const std::string &assets_folder_path, int window_width, int window_height, int nb_tanks);

private:
    std::unique_ptr<PlayerRenderer> player_renderer;
};

#endif//ARENAI_DESKTOP_GAME_ENVIRONMENT_H
