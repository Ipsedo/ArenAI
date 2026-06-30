//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_VISION_H
#define ARENAI_TESTS_VISION_H

#include <gtest/gtest.h>

typedef uint32_t VisionWidth;
typedef uint32_t VisionHeight;
typedef uint32_t VisionChannel;

typedef std::vector<int> OutputConvChannels;
typedef std::vector<int> GroupNormNums;

typedef uint32_t BatchSize;

class VisionTestParam
    : public testing::TestWithParam<std::tuple<
          VisionWidth, VisionHeight, VisionChannel, OutputConvChannels, GroupNormNums, BatchSize>> {
};

#endif//ARENAI_TESTS_VISION_H
