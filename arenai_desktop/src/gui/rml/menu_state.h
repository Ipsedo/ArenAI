//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_GUI_RML_MENU_STATE_H
#define ARENAI_DESKTOP_GUI_RML_MENU_STATE_H

#include "../../core/settings.h"
#include "../gui.h"

namespace arenai::desktop::gui {

    // The state the menu views share: the settings being edited plus the
    // few derived flags more than one view displays. Each view binds the
    // slice it shows into its own data model and refreshes it on show().
    struct MenuState {
        GameSettings settings;
        // the dry-run load of the selected agent succeeded (the Play gate)
        bool agent_valid = false;
        // shown by the pause and game-over popups
        MatchStatus match;
    };

}// namespace arenai::desktop::gui

#endif// ARENAI_DESKTOP_GUI_RML_MENU_STATE_H
