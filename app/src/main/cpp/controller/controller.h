//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_CONTROLLER_H
#define PHYVR_CONTROLLER_H

#include <memory>
#include <string>
#include <map>
#include <android/input.h>

#include "./inputs.h"

class Controller {
public:
    virtual void on_input(const user_input &input) = 0;
};

class ControllerEngine {
public:
    void add_controller(const std::string &name, const std::shared_ptr<Controller> &controller);

    void remove_controller(const std::string &name);

    int32_t on_event(AInputEvent *event);

private:
    std::map<std::string, std::shared_ptr<Controller>> controllers;
};

#endif //PHYVR_CONTROLLER_H
