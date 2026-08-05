//
// Created by samuel on 06/10/2025.
//

#ifndef ARENAI_ENEMY_HANDLER_H
#define ARENAI_ENEMY_HANDLER_H

#include <chrono>

#include <arenai_controller/handler.h>

#include "./types.h"

namespace arenai::core {
    class EnemyControllerHandler final : public controller::EventHandler<Action> {
    public:
        explicit EnemyControllerHandler(
            float refresh_frequency, float wanted_fire_frequency, float turret_rad_per_second);

    protected:
        std::tuple<bool, controller::user_input> to_output(Action event) override;

    private:
        int nb_frames_to_fire;
        int curr_frame;

        float turret_scale_per_frame;
    };
}// namespace arenai::core

#endif//ARENAI_ENEMY_HANDLER_H
