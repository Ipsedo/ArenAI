//
// Created by samuel on 02/06/18.
//

#ifndef PHYVR_PLAYER_H
#define PHYVR_PLAYER_H

#include "base.h"
#include "../graphics/camera.h"
#include "../controls/controls.h"
#include "shooter.h"

class Player : public Base, public Camera, public Controls, public Shooter {

};

#endif //PHYVR_PLAYER_H
