//
// Created by samuel on 19/03/2023.
//

#ifndef PHYVR_CONTROLLER_H
#define PHYVR_CONTROLLER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./inputs.h"

class Controller {
public:
    virtual ~Controller() = default;

    virtual void on_input(const user_input &input) = 0;
};

#endif// PHYVR_CONTROLLER_H
