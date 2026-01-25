//
// Created by samuel on 20/10/2025.
//

#ifndef ARENAI_TRAIN_HOST_ACTION_STATS_H
#define ARENAI_TRAIN_HOST_ACTION_STATS_H

#include <arenai_controller/inputs.h>

#include "./types.h"

class ActionStats {
public:
    ActionStats();

    bool has_fire() const;
    float energy_consumed() const;
    void process_input(const Action &action);

private:
    bool has_fire_;
    float energy_consumed_;
};

#endif//ARENAI_TRAIN_HOST_ACTION_STATS_H
