// TEMPORARY diagnostic probe - to be deleted
#include <iostream>

#include <agents/ppo/ppo_agent.h>
#include <agents/sac/sac_agent.h>
#include <distributions/beta_law.h>
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

TEST(ProbeActSample, SacStatsBiasedMean) {
    torch::manual_seed(1234);
    constexpr int h = 8, w = 8, nb_sensors = 4, nb_cont = 2, nb_disc = 2;

    const auto actor = std::make_shared<Actor>(
        h, w, nb_sensors, nb_cont, nb_disc, 8, std::vector{16},
        std::vector<std::tuple<int, int>>{{3, 4}}, std::vector{2}, 0.3f, 0.1f);

    // push the Beta mean away from 0 : raise alpha, lower beta
    {
        torch::NoGradGuard guard;
        for (auto &p: actor->named_parameters()) {
            if (p.key().find("bias") == std::string::npos) continue;
            if (p.key().find("alpha") != std::string::npos)
                p.value().copy_(torch::tensor({3.f, -1.f}));
            else if (p.key().find("beta") != std::string::npos)
                p.value().copy_(torch::tensor({-1.f, 3.f}));
        }
    }

    const auto agent = std::make_shared<TorchSacAgent>(actor, torch::Device(torch::kCPU));
    const auto state = probe_state(1, h, w, nb_sensors);

    torch::NoGradGuard guard;
    const auto [alpha, beta, disc] = actor->act(state.vision, state.proprioception);
    std::cout << "alpha: " << alpha << "\nbeta: " << beta
              << "\nmean: " << beta_law_mean_action(alpha, beta) << std::endl;

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

    const auto [continuous_action, discrete_action] = agent->act(state, true);
    collector->on_transition(torch::randn({3, 1}), torch::zeros({3, 1}));
    collector->on_episode_end(state);

    torch::NoGradGuard guard;
    const auto [alpha, beta, disc] = actor->act(state.vision, state.proprioception);
    const auto expected_cont = beta_law_log_proba(continuous_action, alpha, beta).sum(-1, true);
    const auto expected_disc =
        (discrete_action * torch::log(torch::clamp(disc, 1e-8, 1.0 - 1e-8))).sum(-1, true);

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
    const auto [alpha, beta, disc] = actor->act(state.vision, state.proprioception);
    std::cout << "alpha: " << alpha << "\nbeta: " << beta << "\ndisc_proba: " << disc << std::endl;

    // deterministic
    const auto [continuous_action, discrete_action] = agent->act(state, false);
    std::cout << "act(false) cont: " << continuous_action
              << "\nact(false) disc: " << discrete_action << std::endl;
    std::cout << "cont == mean ? "
              << torch::allclose(continuous_action, beta_law_mean_action(alpha, beta)) << std::endl;

    // stochastic stats
    constexpr int N = 2000;
    std::vector<torch::Tensor> conts, discs;
    for (int i = 0; i < N; i++) {
        const auto [curr_continuous_action, curr_discrete_action] = agent->act(state, true);
        conts.push_back(curr_continuous_action);
        discs.push_back(curr_discrete_action);
    }
    const auto cont_all = torch::cat(conts, 0);
    const auto disc_all = torch::cat(discs, 0);
    std::cout << "act(true) cont mean: " << cont_all.mean(0)
              << "\nact(true) cont std: " << cont_all.std(0)
              << "\nact(true) cont min: " << std::get<0>(cont_all.min(0))
              << "\nact(true) cont max: " << std::get<0>(cont_all.max(0))
              << "\nact(true) disc freq: " << disc_all.mean(0) << std::endl;
}
