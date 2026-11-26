#include <numbers>

#include <agents/panorama_mask_learner.h>
#include <gtest/gtest.h>
#include <networks/recurrent/liquid_actor.h>
#include <networks/recurrent/liquid_critic.h>

#include <arenai_core/types.h>

using namespace arenai;
using namespace arenai::agent;

namespace {
    constexpr int NB_TANKS = 4;

    std::shared_ptr<PanoramaMaskNetwork> make_encoder() {
        return std::make_shared<PanoramaMaskNetwork>(
            std::vector<std::tuple<int, int>>{{3, 8}, {8, 16}, {16, 16}}, std::vector{2, 4, 4}, 4);
    }

    torch::Tensor random_panorama() {
        return torch::randint(
            255, {NB_TANKS, 3, core::PANORAMA_HEIGHT, core::PANORAMA_WIDTH}, torch::kUInt8);
    }

    std::vector<std::optional<std::vector<EnemySighting>>> one_enemy_ahead_each() {
        return std::vector<std::optional<std::vector<EnemySighting>>>(
            NB_TANKS, std::vector<EnemySighting>{
                          {.bearing = 0.05f,
                           .angular_half_width = 0.01f,
                           .distance = 50.f,
                           .visible = true}});
    }

    std::vector<torch::Tensor> snapshot(const torch::nn::Module &module) {
        std::vector<torch::Tensor> values;
        for (const auto &parameter: module.parameters())
            values.push_back(parameter.detach().clone());
        return values;
    }

    bool unchanged(const torch::nn::Module &module, const std::vector<torch::Tensor> &values) {
        const auto parameters = module.parameters();
        for (size_t i = 0; i < parameters.size(); i++)
            if (!torch::equal(parameters[i], values[i])) return false;
        return true;
    }
}// namespace

TEST(PanoramaMaskLearnerTest, FitLowersTheLossOnARepeatedObservation) {
    const auto encoder = make_encoder();
    PanoramaMaskLearner learner(encoder, 1e-2f, 4, 1, torch::kCPU);
    const auto panorama = random_panorama();
    const auto loss = learner.get_metrics().front();

    learner.observe(panorama, one_enemy_ahead_each());
    learner.fit();
    const float first_loss = loss->last_value();

    for (int i = 0; i < 30; i++) {
        learner.observe(panorama, one_enemy_ahead_each());
        learner.fit();
    }

    ASSERT_LT(loss->last_value(), first_loss / 2.f);
}

TEST(PanoramaMaskLearnerTest, FitWithoutObservationLeavesTheEncoderUntouched) {
    const auto encoder = make_encoder();
    PanoramaMaskLearner learner(encoder, 1e-2f, 4, 1, torch::kCPU);
    const auto before = snapshot(*encoder);

    learner.fit();

    ASSERT_TRUE(unchanged(*encoder, before));
}

TEST(PanoramaMaskLearnerTest, DeadTanksAreNotObserved) {
    const auto encoder = make_encoder();
    PanoramaMaskLearner learner(encoder, 1e-2f, 4, 1, torch::kCPU);
    const auto before = snapshot(*encoder);

    learner.observe(
        random_panorama(),
        std::vector<std::optional<std::vector<EnemySighting>>>(NB_TANKS, std::nullopt));
    learner.fit();

    ASSERT_TRUE(unchanged(*encoder, before));
}

TEST(PanoramaMaskLearnerTest, ActorAndCriticLossesSendNoGradientToTheSharedEncoder) {
    const auto encoder = make_encoder();
    const std::vector<std::tuple<int, int>> vision_channels{{3, 4}};
    const std::vector group_norm_nums{2};

    LiquidActor actor(
        3, 2, 2, 8, vision_channels, group_norm_nums, encoder, 8, 2, 0.1f, 0.1f,
        std::vector(2, 0.2f));
    LiquidCritic critic(3, 8, vision_channels, group_norm_nums, encoder, 8, 2, 0.1f);

    const auto vision =
        torch::randint(255, {NB_TANKS, 3, core::VISION_HEIGHT, core::VISION_WIDTH}, torch::kUInt8);
    const auto panorama = random_panorama();
    const auto sensors = torch::randn({NB_TANKS, 3});

    const auto out = actor.act(vision, panorama, sensors, actor.initial_state(NB_TANKS));
    const auto value = critic.value(vision, panorama, sensors, critic.initial_state(NB_TANKS));
    (out.mu.sum() + out.sigma.sum() + out.discrete.sum() + value.value.sum()).backward();

    for (const auto &parameter: encoder->parameters()) ASSERT_FALSE(parameter.grad().defined());
}

TEST(PanoramaMaskLearnerTest, ActorAndCriticReadTheSameEncoderWeights) {
    const auto encoder = make_encoder();
    const std::vector<std::tuple<int, int>> vision_channels{{3, 4}};
    const std::vector group_norm_nums{2};

    LiquidActor actor(
        3, 2, 2, 8, vision_channels, group_norm_nums, encoder, 8, 2, 0.1f, 0.1f,
        std::vector(2, 0.2f));
    LiquidCritic critic(3, 8, vision_channels, group_norm_nums, encoder, 8, 2, 0.1f);

    const auto actor_weight =
        actor.named_parameters()["panorama_encoder.to_columns.weight"].data_ptr<float>();
    const auto critic_weight =
        critic.named_parameters()["panorama_encoder.to_columns.weight"].data_ptr<float>();

    ASSERT_EQ(actor_weight, critic_weight);
}
