#ifndef ARENAI_TESTS_TORCH_TYPES_H
#define ARENAI_TESTS_TORCH_TYPES_H

#include <gtest/gtest.h>

class TorchTypesTest : public testing::Test {};

typedef int BatchSize;
typedef int VisionHeight;
typedef int VisionWidth;

class TensorToActionsParamTest : public testing::TestWithParam<BatchSize> {};
class StatesToTensorParamTest : public testing::TestWithParam<BatchSize> {};

#endif//ARENAI_TESTS_TORCH_TYPES_H
