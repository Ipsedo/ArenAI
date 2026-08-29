//
// Created by claude on 22/07/2026.
//

#include "./ppo_agent.h"

#include "../../distributions/beta_law.h"
#include "../../distributions/multinomial.h"
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
            const auto &[alpha, beta, discrete_proba] = actor->act(vision, sensors);

            if (sample) {
                action.continuous_action = beta_law_sample(alpha, beta);
                action.discrete_action = multinomial_sample(discrete_proba);
            } else {
                action.continuous_action = beta_law_mean_action(alpha, beta);
                action.discrete_action = multinomial_max_action(discrete_proba);
            }

            // old log-probabilities, kept for the PPO importance ratio
            continuous_log_prob =
                beta_law_log_proba(action.continuous_action, alpha, beta).sum(-1, true);

            const auto clamped_proba = torch::clamp(discrete_proba, EPSILON, 1.0 - EPSILON);
            discrete_log_prob = (action.discrete_action * torch::log(clamped_proba)).sum(-1, true);
        }

        if (collector.has_value())
            collector.value()->on_act(state, action, continuous_log_prob, discrete_log_prob);

        return action;
    }

    void TorchPpoAgent::load(const std::filesystem::path &agent_folder) {
        load_torch(agent_folder, actor, "actor.pt");
    }

}// namespace arenai::agent
