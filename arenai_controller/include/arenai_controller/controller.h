//
// Created by samuel on 19/03/2023.
//

#ifndef ARENAI_CONTROLLER_H
#define ARENAI_CONTROLLER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./inputs.h"

namespace arenai::controller {

    class Controller {
    public:
        virtual ~Controller() = default;

        virtual void apply_input(const user_input &input) = 0;
    };

}// namespace arenai::controller

#endif// ARENAI_CONTROLLER_H
