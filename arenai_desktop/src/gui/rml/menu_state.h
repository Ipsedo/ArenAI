#ifndef ARENAI_DESKTOP_GUI_RML_MENU_STATE_H
#define ARENAI_DESKTOP_GUI_RML_MENU_STATE_H

#include "../../core/settings.h"
#include "../gui.h"

namespace arenai::desktop::gui {

    struct MenuState {
        GameSettings settings;
        // the dry-run load of the selected agent succeeded (the Play gate)
        bool agent_valid = false;
        // shown by the pause and game-over popups
        MatchStatus match;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_MENU_STATE_H
