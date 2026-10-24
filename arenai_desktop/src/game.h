//
// Created by samuel on 18/03/2026.
//

#ifndef ARENAI_DESKTOP_GAME_H
#define ARENAI_DESKTOP_GAME_H

#include "./core/game_options.h"

namespace arenai::desktop {

    // the application: window and gui setup, then the menu / game cycle
    // until the player quits
    void run_gui(const GameOptions &game_options);

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_GAME_H
