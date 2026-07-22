//
// Created by samuel on 21/01/2026.
//

#include "./sac.h"

#include "../../distributions/multinomial.h"
#include "../../distributions/truncated_normal.h"
#include "../../networks_io/torch_loader.h"
#include "../../networks_utils/torch_converter.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

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
            act(states_to_tensor(states, vision_height, vision_width));
        return tensor_to_actions(continuous_action, discrete_action);
    }

    TorchAction TorchSacAgent::act(const TorchState &state) {
        TorchAction action;

        {
            torch::NoGradGuard guard;

            const auto &[vision, sensors] = state;
            const auto &[mu, sigma, discrete_proba] = actor->act(vision, sensors);

            action.continuous_action = truncated_normal_sample(mu, sigma);
            action.discrete_action = multinomial_sample(discrete_proba);
        }

        if (collector.has_value()) collector.value()->on_act(state, action);

        return action;
    }

    void TorchSacAgent::load(const std::filesystem::path &agent_folder) {
        load_torch(agent_folder, actor, "actor.pt");
    }

}// namespace arenai::train
