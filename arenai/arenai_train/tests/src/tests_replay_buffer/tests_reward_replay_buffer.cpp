//
// Created by claude on 01/07/2026.
//

#include <replay_buffer/reward_replay_buffer.h>
#include <reward_transforms/identity_transform.h>
#include <reward_transforms/running_norm.h>
#include <reward_transforms/scale_potential.h>

#include <arenai_train_tests/tests_replay_buffer/tests_reward_replay_buffer.h>

#include "./create_random_step.h"

namespace {
    void fill_buffer_with_known_rewards(
        RewardTransformReplayBuffer &buffer, const int n, const float reward_val,
        const float potential_val) {

        for (int i = 0; i < n; i++) {
            TorchStep step;
            step.state.vision = torch::randint(255, {3, 8, 8}, torch::kUInt8);
            step.state.proprioception = torch::randn({5});
            step.action.continuous_action = torch::randn({3});
            step.action.discrete_action = torch::zeros({2});
            step.action.discrete_action[0] = 1.0f;
            step.reward = torch::tensor({reward_val});
            step.potential = torch::tensor({potential_val});
            step.done = torch::tensor({0.0f});
            step.next_state.vision = torch::randint(255, {3, 8, 8}, torch::kUInt8);
            step.next_state.proprioception = torch::randn({5});

            buffer.add(step);
        }
    }
}// namespace

TEST_F(RewardReplayBufferTest, IdentityTransformsSumRewardAndPotential) {
    auto reward_transform = std::make_shared<IdentityTransform>();
    auto potential_transform = std::make_shared<IdentityTransform>();

    RewardTransformReplayBuffer buffer(20, reward_transform, potential_transform);

    fill_buffer_with_known_rewards(buffer, 10, 2.0f, 3.0f);

    const auto output = buffer.sample(5, torch::kCPU);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(output.reward[i].item<float>(), 5.0f, 1e-4f)
            << "With identity transforms, output reward should be reward + potential = 5.0";
    }
}

TEST_F(RewardReplayBufferTest, ScaleTransformAppliedToReward) {
    constexpr float scale = 2.0f;
    auto reward_transform = std::make_shared<ScalePotentialTransform>(scale);
    auto potential_transform = std::make_shared<IdentityTransform>();

    RewardTransformReplayBuffer buffer(20, reward_transform, potential_transform);

    fill_buffer_with_known_rewards(buffer, 10, 3.0f, 1.0f);

    const auto output = buffer.sample(5, torch::kCPU);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(output.reward[i].item<float>(), 7.0f, 1e-4f)
            << "Output should be scale*reward + potential = 2*3 + 1 = 7";
    }
}

TEST_F(RewardReplayBufferTest, ScaleTransformAppliedToPotential) {
    constexpr float scale = 0.5f;
    auto reward_transform = std::make_shared<IdentityTransform>();
    auto potential_transform = std::make_shared<ScalePotentialTransform>(scale);

    RewardTransformReplayBuffer buffer(20, reward_transform, potential_transform);

    fill_buffer_with_known_rewards(buffer, 10, 1.0f, 4.0f);

    const auto output = buffer.sample(5, torch::kCPU);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(output.reward[i].item<float>(), 3.0f, 1e-4f)
            << "Output should be reward + scale*potential = 1 + 0.5*4 = 3";
    }
}

TEST_F(RewardReplayBufferTest, NormTransformUpdatedOnAdd) {
    auto reward_transform = std::make_shared<NormalizedRewardTransform>(10, 1.0f);
    auto potential_transform = std::make_shared<IdentityTransform>();

    RewardTransformReplayBuffer buffer(20, reward_transform, potential_transform);

    for (int i = 0; i < 10; i++) {
        const auto reward_val = static_cast<float>(i);
        fill_buffer_with_known_rewards(buffer, 1, reward_val, 0.0f);
    }

    const auto output = buffer.sample(5, torch::kCPU);

    ASSERT_TRUE(torch::all(torch::isfinite(output.reward)).item<bool>())
        << "Normalized rewards should be finite";
}

TEST_F(RewardReplayBufferTest, OutputHasCorrectShapes) {
    auto reward_transform = std::make_shared<IdentityTransform>();
    auto potential_transform = std::make_shared<IdentityTransform>();

    RewardTransformReplayBuffer buffer(20, reward_transform, potential_transform);

    fill_buffer_with_known_rewards(buffer, 10, 1.0f, 1.0f);

    const auto output = buffer.sample(4, torch::kCPU);

    ASSERT_EQ(output.state.vision.size(0), 4);
    ASSERT_EQ(output.action.continuous_action.size(0), 4);
    ASSERT_EQ(output.reward.size(0), 4);
    ASSERT_EQ(output.done.size(0), 4);
    ASSERT_EQ(output.next_state.vision.size(0), 4);
}
