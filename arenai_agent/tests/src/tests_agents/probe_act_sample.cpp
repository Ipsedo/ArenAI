// TEMPORARY diagnostic probe - to be deleted
#include <iostream>

#include <agents/ppo/ppo_agent.h>
#include <agents/sac/sac_agent.h>
#include <distributions/truncated_normal.h>
#include <gtest/gtest.h>

using namespace arenai;
using namespace arenai::agent;

namespace {

    TorchState probe_state(const int batch, const int h, const int w, const int nb_sensors) {
        torch::manual_seed(42);
        return {
            .vision = torch::randint(0, 255, {batch, 3, h, w}, torch::kUInt8),
            .proprioception = torch::randn({batch, nb_sensors})};
    }

}// namespace

TEST(ProbeActSample, SacStatsBiasedMu) {
    torch::manual_seed(1234);
    constexpr int h = 8, w = 8, nb_sensors = 4, nb_cont = 2, nb_disc = 2;

    const auto actor = std::make_shared<Actor>(
        h, w, nb_sensors, nb_cont, nb_disc, 8, std::vector{16},
        std::vector<std::tuple<int, int>>{{3, 4}}, std::vector{2}, 0.3f, 0.1f);

    // push mu away from 0 : mu ~ tanh(bias)
    {
        torch::NoGradGuard guard;
        for (auto &p: actor->named_parameters())
            if (p.key().find("mu") != std::string::npos
                && p.key().find("bias") != std::string::npos)
                p.value().copy_(torch::tensor({std::atanh(0.7f), std::atanh(-0.4f)}));
    }

    const auto agent = std::make_shared<TorchSacAgent>(actor, torch::Device(torch::kCPU));
    const auto state = probe_state(1, h, w, nb_sensors);

    torch::NoGradGuard guard;
    const auto [mu, sigma, disc] = actor->act(state.vision, state.proprioception);
    std::cout << "mu: " << mu << "\nsigma: " << sigma << std::endl;

    constexpr int N = 4000;
    std::vector<torch::Tensor> conts;
    for (int i = 0; i < N; i++) conts.push_back(agent->act(state, true).continuous_action);
    const auto cont_all = torch::cat(conts, 0);
    std::cout << "act(true) cont mean: " << cont_all.mean(0)
              << "\nact(true) cont std: " << cont_all.std(0) << std::endl;
}

TEST(ProbeActSample, PpoLogProbs) {
    torch::manual_seed(99);
    constexpr int h = 8, w = 8, nb_sensors = 4, nb_cont = 2, nb_disc = 2;

    const auto actor = std::make_shared<Actor>(
        h, w, nb_sensors, nb_cont, nb_disc, 8, std::vector{16},
        std::vector<std::tuple<int, int>>{{3, 4}}, std::vector{2}, 0.3f, 0.2f);
    const auto rollout_buffer = std::make_shared<PpoRolloutBuffer>();
    const auto collector = std::make_shared<PpoStepCollector>(rollout_buffer);
    const auto agent =
        std::make_shared<TorchPpoAgent>(actor, torch::Device(torch::kCPU), collector);

    const auto state = probe_state(3, h, w, nb_sensors);

    const auto action = agent->act(state, true);
    collector->on_transition(torch::randn({3, 1}), torch::zeros({3, 1}));
    collector->on_episode_end(state);

    torch::NoGradGuard guard;
    const auto [mu, sigma, disc] = actor->act(state.vision, state.proprioception);
    const auto expected_cont =
        truncated_normal_log_pdf(action.continuous_action, mu, sigma).sum(-1, true);
    const auto expected_disc =
        (action.discrete_action * torch::log(torch::clamp(disc, 1e-8, 1.0 - 1e-8))).sum(-1, true);

    const auto rollout = rollout_buffer->get_rollout();
    std::cout << "stored cont lp: " << rollout.continuous_log_probs
              << "\nexpected cont lp: " << expected_cont
              << "\nstored disc lp: " << rollout.discrete_log_probs
              << "\nexpected disc lp: " << expected_disc << std::endl;
}

TEST(ProbeActSample, SacStats) {
    torch::manual_seed(1234);
    constexpr int h = 8, w = 8, nb_sensors = 4, nb_cont = 3, nb_disc = 2;

    const auto actor = std::make_shared<Actor>(
        h, w, nb_sensors, nb_cont, nb_disc, 8, std::vector{16},
        std::vector<std::tuple<int, int>>{{3, 4}}, std::vector{2}, 0.4f, 0.1f);
    const auto agent = std::make_shared<TorchSacAgent>(actor, torch::Device(torch::kCPU));

    const auto state = probe_state(1, h, w, nb_sensors);

    // raw actor output
    torch::NoGradGuard guard;
    const auto [mu, sigma, disc] = actor->act(state.vision, state.proprioception);
    std::cout << "mu: " << mu << "\nsigma: " << sigma << "\ndisc_proba: " << disc << std::endl;

    // deterministic
    const auto det = agent->act(state, false);
    std::cout << "act(false) cont: " << det.continuous_action
              << "\nact(false) disc: " << det.discrete_action << std::endl;
    std::cout << "cont == mu ? " << torch::allclose(det.continuous_action, mu) << std::endl;

    // stochastic stats
    constexpr int N = 2000;
    std::vector<torch::Tensor> conts, discs;
    for (int i = 0; i < N; i++) {
        const auto a = agent->act(state, true);
        conts.push_back(a.continuous_action);
        discs.push_back(a.discrete_action);
    }
    const auto cont_all = torch::cat(conts, 0);
    const auto disc_all = torch::cat(discs, 0);
    std::cout << "act(true) cont mean: " << cont_all.mean(0)
              << "\nact(true) cont std: " << cont_all.std(0)
              << "\nact(true) cont min: " << std::get<0>(cont_all.min(0))
              << "\nact(true) cont max: " << std::get<0>(cont_all.max(0))
              << "\nact(true) disc freq: " << disc_all.mean(0) << std::endl;
}
