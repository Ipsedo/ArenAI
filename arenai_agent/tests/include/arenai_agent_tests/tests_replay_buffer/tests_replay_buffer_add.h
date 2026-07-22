//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TEST_REPLAY_BUFFER_ADD_H
#define ARENAI_TEST_REPLAY_BUFFER_ADD_H

#include "./replay_buffer_test_param.h"

typedef uint32_t StepsNbToAdd;

class ReplayBufferAddNormalTestParam : public ReplayBufferTestParam<StepsNbToAdd> {};
class ReplayBufferAddOverflowTestParam : public ReplayBufferTestParam<StepsNbToAdd> {};

#endif//ARENAI_TEST_REPLAY_BUFFER_ADD_H
