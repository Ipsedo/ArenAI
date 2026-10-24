//
// Created by samuel on 25/10/2026.
//

#ifndef ARENAI_DESKTOP_CONTROLLER_GAMEPAD_SELECTION_H
#define ARENAI_DESKTOP_CONTROLLER_GAMEPAD_SELECTION_H

#include <arenai_view/window.h>

#include "../core/settings.h"

namespace arenai::desktop {

    // routes the window's gamepad input to the device persisted in the
    // settings when it is connected (else the window keeps its fallback)
    void select_configured_gamepad(view::AbstractWindow &window, const GameSettings &settings);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CONTROLLER_GAMEPAD_SELECTION_H
