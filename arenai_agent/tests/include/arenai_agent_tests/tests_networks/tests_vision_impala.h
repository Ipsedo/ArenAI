//
// Created by claude on 20/09/2026.
//

#ifndef ARENAI_TESTS_VISION_IMPALA_H
#define ARENAI_TESTS_VISION_IMPALA_H

#include <gtest/gtest.h>

typedef int ImpalaVisionWidth;
typedef int ImpalaVisionHeight;
typedef int ImpalaVisionChannel;

typedef std::vector<int> ImpalaOutputConvChannels;

typedef int ImpalaBatchSize;

class ImpalaVisionTestParam : public testing::TestWithParam<std::tuple<
                                  ImpalaVisionWidth, ImpalaVisionHeight, ImpalaVisionChannel,
                                  ImpalaOutputConvChannels, ImpalaBatchSize>> {};

class ImpalaVisionEdgeTest : public testing::Test {};

#endif//ARENAI_TESTS_VISION_IMPALA_H
