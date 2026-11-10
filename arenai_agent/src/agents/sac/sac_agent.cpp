//
// Created by claude on 08/11/2026.
//

#include "./sac_agent.h"

#include "../../distributions/bernoulli.h"
#include "../../distributions/gaussian_tanh.h"
#include "../../networks_utils/torch_converter.h"
#include "../../networks_utils/torch_loader.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

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

            const auto [mu, sigma, discrete_proba] = actor->act(state.vision, state.proprioception);

            // a fire without effect is not a decision: the stored action never engages it
            const auto effective_proba =
                discrete_proba * discrete_action_mask(state.fire_mask, discrete_proba.size(-1));

            if (sample) {
                action.continuous_action = gaussian_tanh_sample(mu, sigma).first;
                action.discrete_action = bernoulli_sample(effective_proba);
            } else {
                action.continuous_action = torch::tanh(mu);
                action.discrete_action = bernoulli_max_action(effective_proba);
            }
        }

        if (collector.has_value()) collector.value()->on_act(state, action);

        return action;
    }

    void TorchSacAgent::load(const std::filesystem::path &agent_folder) {
        load_torch(agent_folder, actor, "actor.pt");
    }

}// namespace arenai::agent
