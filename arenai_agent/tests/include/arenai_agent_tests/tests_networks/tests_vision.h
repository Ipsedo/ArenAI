//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_VISION_H
#define ARENAI_TESTS_VISION_H

#include <gtest/gtest.h>

typedef int VisionWidth;
typedef int VisionHeight;
typedef int VisionChannel;

typedef std::vector<int> OutputConvChannels;
typedef std::vector<int> GroupNormNums;

typedef int BatchSize;

class VisionTestParam
    : public testing::TestWithParam<std::tuple<
          VisionWidth, VisionHeight, VisionChannel, OutputConvChannels, GroupNormNums, BatchSize>> {
};

#endif//ARENAI_TESTS_VISION_H
