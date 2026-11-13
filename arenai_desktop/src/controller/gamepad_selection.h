#ifndef ARENAI_DESKTOP_CONTROLLER_GAMEPAD_SELECTION_H
#define ARENAI_DESKTOP_CONTROLLER_GAMEPAD_SELECTION_H

#include <arenai_view/window.h>

#include "../core/settings.h"

namespace arenai::desktop {

    void select_configured_gamepad(view::AbstractWindow &window, const GameSettings &settings);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CONTROLLER_GAMEPAD_SELECTION_H
