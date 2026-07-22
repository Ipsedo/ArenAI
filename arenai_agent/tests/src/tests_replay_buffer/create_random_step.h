//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_ADD_RANDOM_STEP_H
#define ARENAI_ADD_RANDOM_STEP_H

#include <agents/sac/replay_buffer.h>
#include <agents/torch_types.h>

arenai::agent::TorchState create_random_state(int width, int height, int nb_sensors);

arenai::agent::SacInputStep create_random_step(
    int width, int height, int nb_cont_actions, int nb_discrete_actions, int nb_sensors, bool done);

#endif//ARENAI_ADD_RANDOM_STEP_H
