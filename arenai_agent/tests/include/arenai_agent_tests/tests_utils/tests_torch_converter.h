//
// Created by samuel on 30/06/2026.
//

#ifndef ARENAI_TESTS_TORCH_CONVERTER_H
#define ARENAI_TESTS_TORCH_CONVERTER_H

#include <gtest/gtest.h>

class TorchConverterTest : public testing::Test {};

typedef int BatchSize;
typedef int VisionHeight;
typedef int VisionWidth;

class TensorToActionsParamTest : public testing::TestWithParam<BatchSize> {};
class StatesToTensorParamTest
    : public testing::TestWithParam<std::tuple<BatchSize, VisionHeight, VisionWidth>> {};

#endif//ARENAI_TESTS_TORCH_CONVERTER_H
