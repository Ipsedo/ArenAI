//
// Created by samuel on 26/06/2026.
//

#ifndef ARENAI_TRAIN_HOST_TEST_PBUFFER_H
#define ARENAI_TRAIN_HOST_TEST_PBUFFER_H

#include <gtest/gtest.h>

struct image_size {
    int width;
    int height;
};

class PBufferParam : public testing::TestWithParam<image_size> {};

class PBufferSpecularParam : public testing::TestWithParam<image_size> {};

class PBufferClearColorParam : public testing::TestWithParam<image_size> {};

class PBufferMultiFrameParam : public testing::TestWithParam<image_size> {};

#endif//ARENAI_TRAIN_HOST_TEST_PBUFFER_H
