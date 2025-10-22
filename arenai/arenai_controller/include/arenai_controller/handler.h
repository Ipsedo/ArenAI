//
// Created by samuel on 06/10/2025.
//

#ifndef ARENAI_HANDLER_H
#define ARENAI_HANDLER_H

#include "./controller.h"

template<typename Input>
class ControllerHandler {
public:
    virtual ~ControllerHandler() = default;

    ControllerHandler() : controllers() {}

    void add_controller(const std::shared_ptr<Controller> &controller) {
        controllers.push_back(controller);
    }

    bool on_event(Input event) {
        auto [used, input] = to_output(event);
        if (used)
            for (auto &ctrl: controllers) ctrl->on_input(input);
        return used;
    }

protected:
    virtual std::tuple<bool, user_input> to_output(Input event) = 0;

private:
    std::vector<std::shared_ptr<Controller>> controllers;
};
#endif//ARENAI_HANDLER_H
