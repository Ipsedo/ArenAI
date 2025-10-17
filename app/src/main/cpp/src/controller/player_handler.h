//
// Created by samuel on 28/09/2025.
//

#ifndef ARENAI_PLAYER_HANDLER_H
#define ARENAI_PLAYER_HANDLER_H

#include <memory>
#include <vector>

#include <android/configuration.h>
#include <android/input.h>

#include <arenai_controller/controller.h>
#include <arenai_controller/handler.h>
#include <arenai_utils/file_reader.h>
#include <arenai_view/hud.h>

#include "./button.h"
#include "./joystick.h"

class PlayerControllerHandler : public ControllerHandler<AInputEvent *> {
public:
    PlayerControllerHandler(AConfiguration *config, int width, int height);

    std::vector<std::unique_ptr<HUDDrawable>>
    get_hud_drawables(const std::shared_ptr<AbstractFileReader> &file_reader);

protected:
    std::tuple<bool, user_input> to_output(AInputEvent *event);

private:
    std::shared_ptr<HUDJoyStick> drive_joystick;
    std::shared_ptr<HUDButton> fire_button;
    std::shared_ptr<ScreenJoyStick> turret_joystick;
};

#endif// ARENAI_PLAYER_HANDLER_H
