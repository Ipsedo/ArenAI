//
// Created by samuel on 31/05/18.
//

#ifndef PHYVR_SHOOTER_H
#define PHYVR_SHOOTER_H

#include <vector>

#include "../base.h"

class Shooter {
public:
    virtual vector<Base *> fire() = 0;
};

#endif// PHYVR_SHOOTER_H
