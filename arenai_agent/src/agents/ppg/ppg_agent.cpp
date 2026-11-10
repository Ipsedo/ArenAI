#include "./ppg_agent.h"

#include "../../distributions/bernoulli.h"
#include "../../distributions/gaussian_tanh.h"
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
        torch::Tensor discrete_log_prob;

        {
            torch::NoGradGuard guard;

            const auto &[mu, sigma, discrete_proba] =
                actor->act(state.get_vision(), state.get_state());

            if (sample) {
                std::tie(action.continuous_action, continuous_pre_tanh) =
                    gaussian_tanh_sample(mu, sigma);
                action.discrete_action = bernoulli_sample(discrete_proba);
            } else {
                continuous_pre_tanh = mu;
                action.continuous_action = gaussian_tanh_mode(mu);
                action.discrete_action = bernoulli_max_action(discrete_proba);
            }

            continuous_log_prob =
                gaussian_tanh_log_pdf(continuous_pre_tanh, mu, sigma).sum(-1, true);

            discrete_log_prob = (bernoulli_log_proba(action.discrete_action, discrete_proba)
                                 * state.get_discrete_action_mask(discrete_proba.size(-1)))
                                    .sum(-1, true);
        }

        if (collector.has_value())
            collector.value()->on_act(
                state, action, continuous_pre_tanh, continuous_log_prob, discrete_log_prob);

        return action;
    }

    void TorchPpgAgent::load(const std::filesystem::path &agent_folder) {
        load_torch(agent_folder, actor, "actor.pt");
    }

}// namespace arenai::agent
