//
// Created by samuel on 30/06/2026.
//

#include <networks_utils/target_update.h>

#include <arenai_train_tests/tests_networks_utils/tests_target_update.h>

using namespace arenai;
using namespace arenai::train;

namespace {
    std::shared_ptr<torch::nn::LinearImpl> make_linear(int in, int out) {
        return std::make_shared<torch::nn::LinearImpl>(in, out);
    }
}// namespace

TEST_F(TargetUpdateTest, HardUpdateCopiesExact) {
    auto from = make_linear(4, 2);
    auto to = make_linear(4, 2);

    // ensure they differ
    torch::nn::init::ones_(to->weight);
    torch::nn::init::uniform_(from->weight, -1.0, 1.0);

    hard_update(to, from);

    for (auto n_p: from->named_parameters()) {
        const auto &name = n_p.key();
        const auto &from_param = n_p.value();
        const auto &to_param = to->named_parameters()[name];

        ASSERT_TRUE(torch::equal(to_param, from_param))
            << "Parameter " << name << " differs after hard_update";
    }
}

TEST_F(TargetUpdateTest, SoftUpdateTau1IsHardUpdate) {
    auto from = make_linear(4, 2);
    auto to = make_linear(4, 2);

    torch::nn::init::ones_(to->weight);
    torch::nn::init::uniform_(from->weight, -1.0, 1.0);

    soft_update(to, from, 1.0f);

    for (auto n_p: from->named_parameters()) {
        const auto &name = n_p.key();
        const auto &from_param = n_p.value();
        const auto &to_param = to->named_parameters()[name];

        ASSERT_TRUE(torch::allclose(to_param, from_param))
            << "tau=1 should copy exactly, parameter " << name << " differs";
    }
}

TEST_F(TargetUpdateTest, SoftUpdateTau0NoChange) {
    auto from = make_linear(4, 2);
    auto to = make_linear(4, 2);

    // save original 'to' params
    std::map<std::string, torch::Tensor> original;
    for (auto n_p: to->named_parameters()) original[n_p.key()] = n_p.value().clone();

    soft_update(to, from, 0.0f);

    for (auto n_p: to->named_parameters()) {
        const auto &name = n_p.key();
        ASSERT_TRUE(torch::allclose(n_p.value(), original[name]))
            << "tau=0 should not change parameter " << name;
    }
}

TEST_F(TargetUpdateTest, SoftUpdateInterpolation) {
    auto from = make_linear(4, 2);
    auto to = make_linear(4, 2);

    constexpr float tau = 0.5f;

    // save original params
    std::map<std::string, torch::Tensor> to_orig;
    for (auto n_p: to->named_parameters()) to_orig[n_p.key()] = n_p.value().clone();

    soft_update(to, from, tau);

    for (auto n_p: from->named_parameters()) {
        const auto &name = n_p.key();
        const auto expected = tau * n_p.value() + (1.0f - tau) * to_orig[name];

        ASSERT_TRUE(torch::allclose(to->named_parameters()[name], expected))
            << "tau=0.5 should average parameters, " << name << " differs";
    }
}
