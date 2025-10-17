//
// Created by samuel on 06/10/2025.
//

#ifndef ARENAI_ENEMY_HANDLER_H
#define ARENAI_ENEMY_HANDLER_H

#include <chrono>

#include <arenai_controller/handler.h>

#include "./types.h"

class EnemyControllerHandler final : public ControllerHandler<Action> {
public:
    explicit EnemyControllerHandler(float refresh_frequency, float wanted_fire_frequency);

protected:
    std::tuple<bool, user_input> to_output(Action event) override;

private:
    int nb_frames_to_fire;
    int curr_frame;
};

#endif//ARENAI_ENEMY_HANDLER_H
