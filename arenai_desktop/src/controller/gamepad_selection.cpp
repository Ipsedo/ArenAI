//
// Created by samuel on 25/10/2026.
//

#include "./gamepad_selection.h"

namespace arenai::desktop {

    void select_configured_gamepad(view::AbstractWindow &window, const GameSettings &settings) {
        const auto &wanted_guid = settings.bindings.gamepad.device_guid;
        if (settings.controller_kind != ControllerKind::Gamepad || wanted_guid.empty()) return;

        for (const auto &[id, name, guid]: window.list_gamepads())
            if (guid == wanted_guid) {
                window.select_gamepad(id);
                return;
            }
    }

}// namespace arenai::desktop
