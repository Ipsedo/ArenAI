//
// Created by samuel on 06/10/2025.
//

#ifndef ARENAI_HANDLER_H
#define ARENAI_HANDLER_H

#include "./controller.h"

namespace arenai::controller {

    template<typename Input>
    class EventHandler {
    public:
        virtual ~EventHandler() = default;

        EventHandler() = default;

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

}// namespace arenai::controller

#endif//ARENAI_HANDLER_H
