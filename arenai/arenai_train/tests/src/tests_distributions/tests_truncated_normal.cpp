//
// Created by samuel on 30/06/2026.
//

#include <distributions/truncated_normal.h>

#include <arenai_train_tests/tests_distributions/tests_truncated_normal.h>

TEST_P(TruncatedNormalTestParam, TruncatedNormalSample) {
    const auto [lower_bound, upper_bound, shape] = GetParam();

    const auto mu = torch::rand(shape, torch::TensorOptions().dtype(torch::kFloat32))
                        * (upper_bound - lower_bound)
                    + lower_bound;
    const auto sigma = torch::rand(shape);

    const auto cont_action = truncated_normal_sample(mu, sigma, lower_bound, upper_bound);

    ASSERT_EQ(cont_action.ndimension(), shape.size());

    for (int i = 0; i < shape.size(); i++) ASSERT_EQ(cont_action.size(i), shape[i]);

    ASSERT_TRUE(
        torch::all(torch::logical_and(
                       torch::ge(cont_action, lower_bound), torch::le(cont_action, upper_bound)))
            .item<bool>());
}

TEST_P(TruncatedNormalTestParam, TruncatedNormalEntropy) {
    const auto [lower_bound, upper_bound, shape] = GetParam();

    const auto mu = torch::rand(shape, torch::TensorOptions().dtype(torch::kFloat32))
                        * (upper_bound - lower_bound)
                    + lower_bound;
    const auto sigma = torch::rand(shape);

    const auto entropy = truncated_normal_entropy(mu, sigma, lower_bound, upper_bound);

    ASSERT_EQ(entropy.ndimension(), shape.size());

    for (int i = 0; i < shape.size(); i++) ASSERT_EQ(entropy.size(i), shape[i]);
}

TEST_P(TruncatedNormalTestParam, TruncatedNormalTargetEntropy) {
    const auto [lower_bound, upper_bound, shape] = GetParam();

    const auto sigma = torch::rand({1}).item<float>();

    ASSERT_NO_THROW(truncated_normal_target_entropy(shape.size(), sigma, lower_bound, upper_bound));
}

INSTANTIATE_TEST_SUITE_P(
    TruncatedNormal, TruncatedNormalTestParam,
    testing::Combine(
        testing::Values(-1.0, -0.5, 0.0), testing::Values(0.5, 0.75, 1.0),
        testing::Values(Shape{1}, Shape{2}, Shape{2, 3})));
