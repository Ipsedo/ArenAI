//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_REPLAY_BUFFER_SAMPLE_H
#define ARENAI_TESTS_REPLAY_BUFFER_SAMPLE_H

#include "./replay_buffer_test_param.h"

typedef uint32_t BatchSize;
typedef uint32_t StepsNbToAdd;

class ReplayBufferSampleNormalTestParam : public ReplayBufferTestParam<BatchSize, StepsNbToAdd> {};
class ReplayBufferSampleOverflowTestParam : public ReplayBufferTestParam<BatchSize, StepsNbToAdd> {
};
class ReplayBufferSampleUnderflowTestParam : public ReplayBufferTestParam<BatchSize, StepsNbToAdd> {
};
class ReplayBufferSampleDoubleOverflowTestParam
    : public ReplayBufferTestParam<BatchSize, StepsNbToAdd> {};

#endif//ARENAI_TESTS_REPLAY_BUFFER_SAMPLE_H
