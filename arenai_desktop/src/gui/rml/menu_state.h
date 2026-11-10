#ifndef ARENAI_DESKTOP_GUI_RML_MENU_STATE_H
#define ARENAI_DESKTOP_GUI_RML_MENU_STATE_H

#include "../../core/settings.h"
#include "../gui.h"

namespace arenai::desktop::gui {

    struct MenuState {
        GameSettings settings;
        bool agent_valid = false;
        MatchStatus match;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_MENU_STATE_H
