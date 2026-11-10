#ifndef ARENAI_TESTS_ACTOR_H
#define ARENAI_TESTS_ACTOR_H

#include <gtest/gtest.h>

typedef std::vector<int> HiddenLayers;
typedef int ContinuousActionsNb;
typedef int DiscreteActionsNb;

typedef int SensorsNb;
typedef int SensorsHiddenSize;

typedef int BatchSize;

class ActorTestParam : public testing::TestWithParam<std::tuple<
                           HiddenLayers, ContinuousActionsNb, DiscreteActionsNb, SensorsNb,
                           SensorsHiddenSize, BatchSize>> {};

#endif//ARENAI_TESTS_ACTOR_H
