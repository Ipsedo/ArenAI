//
// Created by samuel on 21/01/2026.
//

#include "./sac_agent.h"

#include "../../distributions/beta_law.h"
#include "../../distributions/multinomial.h"
#include "../../networks_utils/torch_converter.h"
#include "../../networks_utils/torch_loader.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    /*
     * Torch SAC agent
     */

    TorchSacAgent::TorchSacAgent(
        const std::shared_ptr<Actor> &actor, const torch::Device device,
        std::optional<std::shared_ptr<SacStepCollector>> collector)
        : actor(actor), collector(std::move(collector)) {
        actor->to(device);
    }

    std::vector<core::Action> TorchSacAgent::act(
        const std::vector<core::State> &states, const int vision_height, const int vision_width) {
        const auto [continuous_action, discrete_action] =
            act(states_to_tensor(states, vision_height, vision_width), false);
        return tensor_to_actions(continuous_action, discrete_action);
    }

    TorchAction TorchSacAgent::act(const TorchState &state, const bool sample) {
        TorchAction action;

        {
            torch::NoGradGuard guard;

            actor->train(false);

            const auto &[vision, sensors] = state;
            const auto &[alpha, beta, discrete_proba] = actor->act(vision, sensors);

            if (sample) {
                action.continuous_action = beta_law_sample(alpha, beta);
                action.discrete_action = multinomial_sample(discrete_proba);
            } else {
                action.continuous_action = beta_law_mean_action(alpha, beta);
                action.discrete_action = multinomial_max_action(discrete_proba);
            }
        }

        if (collector.has_value()) collector.value()->on_act(state, action);

        return action;
    }

    void TorchSacAgent::load(const std::filesystem::path &agent_folder) {
        load_torch(agent_folder, actor, "actor.pt");
    }

}// namespace arenai::agent
