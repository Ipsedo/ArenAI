//
// Created by samuel on 06/10/2025.
//

#ifndef PHYVR_ENEMY_HANDLER_H
#define PHYVR_ENEMY_HANDLER_H

#include <chrono>

#include <phyvr_controller/handler.h>

#include "./types.h"

class EnemyControllerHandler : public ControllerHandler<Action> {
public:
    explicit EnemyControllerHandler(float fire_latency_seconds);

protected:
    std::tuple<bool, user_input> to_output(Action event) override;

private:
    float fire_latency_seconds;

    std::chrono::time_point<std::chrono::steady_clock> last_time;
};

#endif//PHYVR_ENEMY_HANDLER_H
