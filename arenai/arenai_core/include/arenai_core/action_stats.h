//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ACTION_STATS_H
#define ARENAI_TRAIN_HOST_ACTION_STATS_H

#include <arenai_controller/inputs.h>

#include "./types.h"

class ActionStats {
public:
    ActionStats(float nb_seconds_idle_tolerated, float wanted_frequency, float delta_action);

    float get_reward() const;
    void process_input(const Action &action);

private:
    int nb_frames_idle;
    float delta_action;

    int frames_since_last_action;
    int max_frames_penalty;

    Action last_action;
};

#endif//ARENAI_TRAIN_HOST_ACTION_STATS_H
