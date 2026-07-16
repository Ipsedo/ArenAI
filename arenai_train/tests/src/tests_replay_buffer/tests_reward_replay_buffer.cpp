//
// Created by claude on 01/07/2026.
//

#include <replay_buffer/reward_replay_buffer.h>
#include <reward_transforms/identity_transform.h>
#include <reward_transforms/running_norm_transform.h>
#include <reward_transforms/scale_transform.h>

#include <arenai_train_tests/tests_replay_buffer/tests_reward_replay_buffer.h>

#include "./create_random_step.h"

using namespace arenai;
using namespace arenai::train;

namespace {
    void fill_buffer_with_known_reward(
        RewardTransformReplayBuffer &buffer, const int n, const float reward_val) {

        for (int i = 0; i < n; i++) {
            TorchInputStep step;
            step.state.vision = torch::randint(255, {3, 8, 8}, torch::kUInt8);
            step.state.proprioception = torch::randn({5});
            step.action.continuous_action = torch::randn({3});
            step.action.discrete_action = torch::zeros({2});
            step.action.discrete_action[0] = 1.0f;
            step.main_reward = torch::tensor({reward_val});
            step.potential_reward = torch::tensor({0.0});
            step.done = torch::tensor({0.0f});
            step.next_state.vision = torch::randint(255, {3, 8, 8}, torch::kUInt8);
            step.next_state.proprioception = torch::randn({5});

            buffer.add(step);
        }
    }
}// namespace

TEST_F(RewardReplayBufferTest, IdentityTransformLeavesRewardUnchanged) {
    const auto reward_transform = std::make_shared<IdentityTransform>();

    RewardTransformReplayBuffer buffer(20, reward_transform, reward_transform);

    fill_buffer_with_known_reward(buffer, 10, 2.0f);

    const auto output = buffer.sample(5, torch::kCPU);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(output.reward[i].item<float>(), 2.0f, 1e-4f)
            << "Identity transform should leave the reward unchanged";
    }
}

TEST_F(RewardReplayBufferTest, ScaleTransformAppliedToReward) {
    constexpr float scale = 2.0f;
    const auto reward_transform = std::make_shared<ScalePotentialTransform>(scale);

    RewardTransformReplayBuffer buffer(20, reward_transform, reward_transform);

    fill_buffer_with_known_reward(buffer, 10, 3.0f);

    const auto output = buffer.sample(5, torch::kCPU);

    for (int i = 0; i < 5; i++) {
        ASSERT_NEAR(output.reward[i].item<float>(), 6.0f, 1e-4f)
            << "Output should be scale*reward = 2*3 = 6";
    }
}

TEST_F(RewardReplayBufferTest, NormTransformProducesFiniteRewards) {
    const auto reward_transform = std::make_shared<NormalizedRewardTransform>(10, 1.0f);

    RewardTransformReplayBuffer buffer(20, reward_transform, reward_transform);

    for (int i = 0; i < 10; i++) {
        const auto reward_val = static_cast<float>(i);
        fill_buffer_with_known_reward(buffer, 1, reward_val);
    }

    const auto output = buffer.sample(5, torch::kCPU);

    ASSERT_TRUE(torch::all(torch::isfinite(output.reward)).item<bool>())
        << "Normalized rewards should be finite";
}

TEST_F(RewardReplayBufferTest, OutputHasCorrectShapes) {
    auto reward_transform = std::make_shared<IdentityTransform>();

    RewardTransformReplayBuffer buffer(20, reward_transform, reward_transform);

    fill_buffer_with_known_reward(buffer, 10, 1.0f);

    const auto [state, action, reward, done, next_state] = buffer.sample(4, torch::kCPU);

    ASSERT_EQ(state.vision.size(0), 4);
    ASSERT_EQ(action.continuous_action.size(0), 4);
    ASSERT_EQ(reward.size(0), 4);
    ASSERT_EQ(done.size(0), 4);
    ASSERT_EQ(next_state.vision.size(0), 4);
}
