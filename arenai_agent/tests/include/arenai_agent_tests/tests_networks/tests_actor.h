//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_ACTOR_H
#define ARENAI_TESTS_ACTOR_H

#include <gtest/gtest.h>

typedef std::vector<int> HiddenLayers;
typedef uint32_t ContinuousActionsNb;
typedef uint32_t DiscreteActionsNb;

typedef uint32_t SensorsNb;
typedef uint32_t SensorsHiddenSize;

typedef uint32_t BatchSize;

class ActorTestParam : public testing::TestWithParam<std::tuple<
                           HiddenLayers, ContinuousActionsNb, DiscreteActionsNb, SensorsNb,
                           SensorsHiddenSize, BatchSize>> {};

#endif//ARENAI_TESTS_ACTOR_H
