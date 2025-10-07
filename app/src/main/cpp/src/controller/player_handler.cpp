//
// Created by samuel on 19/03/2023.
//

#include "./player_handler.h"

#include <android/input.h>
#include <android_native_app_glue.h>

#include <phyvr_controller/controller.h>

PlayerControllerHandler::PlayerControllerHandler(AConfiguration *config, int width, int height)
    : ControllerHandler<AInputEvent *>() {
    drive_joystick = std::make_shared<HUDJoyStick>(config, width, height, 40, 150, 65);
    fire_button = std::make_shared<HUDButton>(config, width, height, 60, 80);
    turret_joystick = std::make_shared<ScreenJoyStick>(
        width, height, std::vector<std::shared_ptr<PointerLocker>>{drive_joystick, fire_button});
}

std::tuple<bool, user_input> PlayerControllerHandler::to_output(AInputEvent *event) {
    int drive_joystick_used = drive_joystick->on_event(event);
    int fire_button_used = fire_button->on_event(event);
    int turret_joystick_used = turret_joystick->on_event(event);

    user_input input{
        drive_joystick->get_input(),
        turret_joystick->get_input(),
        {fire_button->get_input()},
        {false},
        {false}};

    return {drive_joystick_used || fire_button_used || turret_joystick_used, input};
}

std::vector<std::unique_ptr<HUDDrawable>>
PlayerControllerHandler::get_hud_drawables(const std::shared_ptr<AbstractFileReader> &file_reader) {
    std::vector<std::unique_ptr<HUDDrawable>> result{};

    auto left_joystick_drawable = drive_joystick->get_hud_drawable(file_reader);
    result.push_back(std::move(left_joystick_drawable));

    auto fire_button_drawable = fire_button->get_hud_drawable(file_reader);
    result.push_back(std::move(fire_button_drawable));

    return result;
}
