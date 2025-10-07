//
// Created by samuel on 28/09/2025.
//

#ifndef PHYVR_PLAYER_HANDLER_H
#define PHYVR_PLAYER_HANDLER_H

#include <memory>
#include <vector>

#include <android/configuration.h>
#include <android/input.h>

#include <phyvr_controller/controller.h>
#include <phyvr_controller/handler.h>
#include <phyvr_utils/file_reader.h>
#include <phyvr_view/hud.h>

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

#endif// PHYVR_PLAYER_HANDLER_H
