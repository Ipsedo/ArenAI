//
// Created by claude on 22/07/2026.
//

#include "./ppo_agent.h"

#include "../../distributions/bernoulli.h"
#include "../../distributions/beta_law.h"
#include "../../networks/constants.h"
#include "../../networks_utils/torch_converter.h"
#include "../../networks_utils/torch_loader.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    /*
     * Torch PPO agent
     */

    TorchPpoAgent::TorchPpoAgent(
        const std::shared_ptr<Actor> &actor, const torch::Device device,
        std::optional<std::shared_ptr<PpoStepCollector>> collector)
        : actor(actor), collector(std::move(collector)) {
        actor->to(device);
    }

    std::vector<core::Action> TorchPpoAgent::act(
        const std::vector<core::State> &states, const int vision_height, const int vision_width) {
        const auto [continuous_action, discrete_action] =
            act(states_to_tensor(states, vision_height, vision_width), false);
        return tensor_to_actions(continuous_action, discrete_action);
    }

    TorchAction TorchPpoAgent::act(const TorchState &state, const bool sample) {
        TorchAction action;
        torch::Tensor continuous_log_prob;
        torch::Tensor discrete_log_prob;

        {
            torch::NoGradGuard guard;

            const auto &[vision, sensors] = state;
            const auto &[mode, concentration, discrete_proba] = actor->act(vision, sensors);

            if (sample) {
                action.continuous_action = beta_law_sample(mode, concentration);
                action.discrete_action = bernoulli_sample(discrete_proba);
            } else {
                action.continuous_action = beta_law_mode_action(mode);
                action.discrete_action = bernoulli_max_action(discrete_proba);
            }

            // old log-probabilities, kept for the PPO importance ratio
            continuous_log_prob =
                beta_law_log_proba(action.continuous_action, mode, concentration).sum(-1, true);

            discrete_log_prob =
                bernoulli_log_proba(action.discrete_action, discrete_proba).sum(-1, true);
        }

        if (collector.has_value())
            collector.value()->on_act(state, action, continuous_log_prob, discrete_log_prob);

        return action;
    }

    void TorchPpoAgent::load(const std::filesystem::path &agent_folder) {
        load_torch(agent_folder, actor, "actor.pt");
    }

}// namespace arenai::agent
