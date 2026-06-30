//
// Created by samuel on 30/06/2026.
//

#include <arenai_core/constants.h>
#include <arenai_train/torch_converter.h>
#include <arenai_train_tests/tests_utils/tests_torch_converter.h>

// ========================================================================
// Fixed tests
// ========================================================================

TEST_F(TorchConverterTest, SingleActionMapping) {
    const auto continuous = torch::tensor({{0.1f, -0.2f, 0.3f, -0.4f}});
    const auto discrete = torch::tensor({{0.8f, 0.2f}});

    const auto actions = tensor_to_actions(continuous, discrete);

    ASSERT_EQ(actions.size(), 1);
    ASSERT_FLOAT_EQ(actions[0].left_joystick.x, 0.1f);
    ASSERT_FLOAT_EQ(actions[0].left_joystick.y, -0.2f);
    ASSERT_FLOAT_EQ(actions[0].right_joystick.x, 0.3f);
    ASSERT_FLOAT_EQ(actions[0].right_joystick.y, -0.4f);
    ASSERT_TRUE(actions[0].fire_button.pressed);
}

TEST_F(TorchConverterTest, FireButtonFalseWhenSecondLarger) {
    const auto continuous = torch::zeros({1, 4});
    const auto discrete = torch::tensor({{0.2f, 0.8f}});

    const auto actions = tensor_to_actions(continuous, discrete);

    ASSERT_FALSE(actions[0].fire_button.pressed);
}

TEST_F(TorchConverterTest, FireButtonFalseWhenEqual) {
    const auto continuous = torch::zeros({1, 4});
    const auto discrete = torch::tensor({{0.5f, 0.5f}});

    const auto actions = tensor_to_actions(continuous, discrete);

    ASSERT_FALSE(actions[0].fire_button.pressed);
}

TEST_F(TorchConverterTest, SingleStateToTensor) {
    constexpr int H = 4, W = 6;

    State state;
    state.vision.pixels.resize(3 * H * W, 128);
    state.proprioception.resize(ENEMY_PROPRIOCEPTION_SIZE, 1.0f);

    const auto [vision, proprioception] = state_to_tensor(state, H, W);

    ASSERT_EQ(vision.dim(), 3);
    ASSERT_EQ(vision.size(0), 3);
    ASSERT_EQ(vision.size(1), H);
    ASSERT_EQ(vision.size(2), W);
    ASSERT_EQ(vision.dtype(), torch::kUInt8);

    ASSERT_EQ(proprioception.dim(), 1);
    ASSERT_EQ(proprioception.size(0), ENEMY_PROPRIOCEPTION_SIZE);
    ASSERT_EQ(proprioception.dtype(), torch::kFloat);
}

TEST_F(TorchConverterTest, StateVisionPixelsPreserved) {
    constexpr int H = 2, W = 2;

    State state;
    state.vision.pixels = {
        10, 20,  30,  40,// C=0
        50, 60,  70,  80,// C=1
        90, 100, 110, 120// C=2
    };
    state.proprioception.resize(ENEMY_PROPRIOCEPTION_SIZE, 0.0f);

    const auto [vision, proprioception] = state_to_tensor(state, H, W);

    auto acc = vision.accessor<uint8_t, 3>();
    ASSERT_EQ(acc[0][0][0], 10);
    ASSERT_EQ(acc[0][0][1], 20);
    ASSERT_EQ(acc[0][1][0], 30);
    ASSERT_EQ(acc[0][1][1], 40);
    ASSERT_EQ(acc[1][0][0], 50);
    ASSERT_EQ(acc[2][1][1], 120);
}

TEST_F(TorchConverterTest, StateProprioceptionPreserved) {
    constexpr int H = 2, W = 2;

    State state;
    state.vision.pixels.resize(3 * H * W, 0);
    state.proprioception.resize(ENEMY_PROPRIOCEPTION_SIZE);
    for (int i = 0; i < ENEMY_PROPRIOCEPTION_SIZE; ++i)
        state.proprioception[i] = static_cast<float>(i) * 0.1f;

    const auto [vision, proprioception] = state_to_tensor(state, H, W);

    auto acc = proprioception.accessor<float, 1>();
    for (int i = 0; i < ENEMY_PROPRIOCEPTION_SIZE; ++i)
        ASSERT_FLOAT_EQ(acc[i], static_cast<float>(i) * 0.1f);
}

// ========================================================================
// Parameterized: tensor_to_actions batch sizes
// ========================================================================

TEST_P(TensorToActionsParamTest, BatchSizePreserved) {
    const auto batch = GetParam();

    const auto continuous = torch::randn({batch, 4});
    const auto discrete = torch::rand({batch, 2});

    const auto actions = tensor_to_actions(continuous, discrete);

    ASSERT_EQ(static_cast<int>(actions.size()), batch);
}

TEST_P(TensorToActionsParamTest, ContinuousValuesMatchTensor) {
    const auto batch = GetParam();

    const auto continuous = torch::randn({batch, 4});
    const auto discrete = torch::rand({batch, 2});

    const auto actions = tensor_to_actions(continuous, discrete);

    auto acc = continuous.accessor<float, 2>();
    for (int i = 0; i < batch; ++i) {
        ASSERT_FLOAT_EQ(actions[i].left_joystick.x, acc[i][0]);
        ASSERT_FLOAT_EQ(actions[i].left_joystick.y, acc[i][1]);
        ASSERT_FLOAT_EQ(actions[i].right_joystick.x, acc[i][2]);
        ASSERT_FLOAT_EQ(actions[i].right_joystick.y, acc[i][3]);
    }
}

TEST_P(TensorToActionsParamTest, DiscreteFireButtonConsistent) {
    const auto batch = GetParam();

    const auto discrete = torch::rand({batch, 2});
    const auto continuous = torch::zeros({batch, 4});

    const auto actions = tensor_to_actions(continuous, discrete);

    auto acc = discrete.accessor<float, 2>();
    for (int i = 0; i < batch; ++i) {
        ASSERT_EQ(actions[i].fire_button.pressed, acc[i][0] > acc[i][1]);
    }
}

TEST_P(TensorToActionsParamTest, WorksWithGPUTensor) {
    const auto batch = GetParam();

    auto continuous = torch::randn({batch, 4});
    auto discrete = torch::rand({batch, 2});

    if (torch::cuda::is_available()) {
        continuous = continuous.cuda();
        discrete = discrete.cuda();
    }

    const auto actions = tensor_to_actions(continuous, discrete);

    ASSERT_EQ(static_cast<int>(actions.size()), batch);
}

INSTANTIATE_TEST_SUITE_P(
    TorchConverter, TensorToActionsParamTest, testing::Values(1, 2, 4, 8, 16, 32));

// ========================================================================
// Parameterized: states_to_tensor batch × vision dimensions
// ========================================================================

TEST_P(StatesToTensorParamTest, OutputShapeCorrect) {
    const auto [batch, height, width] = GetParam();

    std::vector<State> states(batch);
    for (auto &s: states) {
        s.vision.pixels.resize(3 * height * width, 0);
        s.proprioception.resize(ENEMY_PROPRIOCEPTION_SIZE, 0.0f);
    }

    const auto [vision, proprioception] = states_to_tensor(states, height, width);

    ASSERT_EQ(vision.size(0), batch);
    ASSERT_EQ(vision.size(1), 3);
    ASSERT_EQ(vision.size(2), height);
    ASSERT_EQ(vision.size(3), width);
    ASSERT_EQ(vision.dtype(), torch::kUInt8);

    ASSERT_EQ(proprioception.size(0), batch);
    ASSERT_EQ(proprioception.size(1), ENEMY_PROPRIOCEPTION_SIZE);
    ASSERT_EQ(proprioception.dtype(), torch::kFloat);
}

TEST_P(StatesToTensorParamTest, NoGradient) {
    const auto [batch, height, width] = GetParam();

    std::vector<State> states(batch);
    for (auto &s: states) {
        s.vision.pixels.resize(3 * height * width, 0);
        s.proprioception.resize(ENEMY_PROPRIOCEPTION_SIZE, 0.0f);
    }

    const auto [vision, proprioception] = states_to_tensor(states, height, width);

    ASSERT_FALSE(vision.requires_grad());
    ASSERT_FALSE(proprioception.requires_grad());
}

TEST_P(StatesToTensorParamTest, PixelDataPreserved) {
    const auto [batch, height, width] = GetParam();

    std::vector<State> states(batch);
    for (int b = 0; b < batch; ++b) {
        states[b].vision.pixels.resize(3 * height * width);
        for (int i = 0; i < 3 * height * width; ++i)
            states[b].vision.pixels[i] = static_cast<uint8_t>((b + i) % 256);
        states[b].proprioception.resize(ENEMY_PROPRIOCEPTION_SIZE, 0.0f);
    }

    const auto [vision, proprioception] = states_to_tensor(states, height, width);

    auto acc = vision.accessor<uint8_t, 4>();
    for (int b = 0; b < batch; ++b) {
        int idx = 0;
        for (int c = 0; c < 3; ++c)
            for (int h = 0; h < height; ++h)
                for (int w = 0; w < width; ++w, ++idx)
                    ASSERT_EQ(acc[b][c][h][w], static_cast<uint8_t>((b + idx) % 256));
    }
}

INSTANTIATE_TEST_SUITE_P(
    TorchConverter, StatesToTensorParamTest,
    testing::Combine(
        testing::Values(1, 2, 4, 8), testing::Values(2, 4, 8), testing::Values(2, 4, 8)));
