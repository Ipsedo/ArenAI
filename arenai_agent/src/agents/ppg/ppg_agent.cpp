#include "./ppg_agent.h"

#include "../../distributions/bernoulli.h"
#include "../../distributions/gaussian_tanh.h"
#include "../../distributions/spatial_aim_law.h"
#include "../../networks/constants.h"
#include "../../networks_utils/torch_loader.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    TorchPpgAgent::TorchPpgAgent(
        const std::shared_ptr<DistillationActor> &actor, const torch::Device device,
        std::optional<std::shared_ptr<PpgStepCollector>> collector)
        : actor(actor), collector(std::move(collector)) {
        actor->to(device);
    }

    std::vector<core::Action> TorchPpgAgent::act(
        const std::vector<core::State> &states, const int vision_height, const int vision_width) {
        return act(TorchState(states, vision_height, vision_width), false).to_actions();
    }

    TorchAction TorchPpgAgent::act(const TorchState &state, const bool sample) {
        TorchAction action;
        torch::Tensor continuous_pre_tanh;
        torch::Tensor continuous_log_prob;
        AimDraw aim_draw;
        torch::Tensor aim_log_prob;
        torch::Tensor discrete_log_prob;

        {
            torch::NoGradGuard guard;

            const auto out = actor->act(state.get_vision(), state.get_state());
            const auto &mu = out.mu;
            const auto &sigma = out.sigma;
            const auto &discrete_proba = out.discrete;

            torch::Tensor drive;
            if (sample) {
                std::tie(drive, continuous_pre_tanh) = gaussian_tanh_sample(mu, sigma);
                aim_draw = out.aim.sample();
                action.discrete_action = bernoulli_sample(discrete_proba);
            } else {
                continuous_pre_tanh = mu;
                drive = gaussian_tanh_mode(mu);
                aim_draw = out.aim.mode();
                action.discrete_action = bernoulli_max_action(discrete_proba);
            }
            action.continuous_action = torch::cat({drive, out.aim.point(aim_draw)}, -1);
            aim_log_prob = out.aim.log_prob(aim_draw);

            // old log-probabilities per continuous action, kept for the PPG importance ratios
            continuous_log_prob = gaussian_tanh_log_pdf(continuous_pre_tanh, mu, sigma);

            // a fire without effect is not a decision: no log-probability, no gradient
            discrete_log_prob = (bernoulli_log_proba(action.discrete_action, discrete_proba)
                                 * state.get_discrete_action_mask(discrete_proba.size(-1)))
                                    .sum(-1, true);
        }

        if (collector.has_value())
            collector.value()->on_act(
                state, action, continuous_pre_tanh, continuous_log_prob, aim_draw.cell,
                aim_draw.pre_tanh, aim_log_prob, discrete_log_prob);

        return action;
    }

    void TorchPpgAgent::load(const std::filesystem::path &agent_folder) {
        load_torch(agent_folder, actor, "actor.pt");
    }

}// namespace arenai::agent
