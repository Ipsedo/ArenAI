//
// Created by claude on 21/07/2026.
//

#ifndef ARENAI_TORCH_TYPES_H
#define ARENAI_TORCH_TYPES_H

#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include <arenai_core/types.h>

namespace arenai::agent {

    class TorchState {
    public:
        TorchState() = default;

        TorchState(const std::vector<core::State> &states, int vision_height, int vision_width);
        TorchState(torch::Tensor vision, torch::Tensor state, torch::Tensor fire_mask);

        const torch::Tensor &get_vision() const;
        const torch::Tensor &get_state() const;
        const torch::Tensor &get_fire_mask() const;

        std::vector<float> get_remaining_shells_ratio() const;
        std::vector<float> get_cooldown_ratio() const;

        torch::Tensor get_discrete_action_mask(int64_t nb_discrete_actions) const;
        static torch::Tensor
        discrete_action_mask(const torch::Tensor &fire_mask, int64_t nb_discrete_actions);

        TorchState to(torch::Device device) const;
        TorchState detach() const;

        TorchState slice(int64_t start, int64_t end) const;

    private:
        torch::Tensor vision;
        torch::Tensor state;
        torch::Tensor fire_mask;
    };

    struct TorchAction {
        torch::Tensor continuous_action;
        torch::Tensor discrete_action;

        std::vector<core::Action> to_actions() const;

        static const std::vector<std::string> &continuous_action_names();
        static const std::vector<std::string> &discrete_action_names();
    };

    class TorchStep {
    public:
        TorchStep(
            const std::vector<
                std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>> &steps,
            int vision_height, int vision_width);

        const TorchState &get_states() const;

        const torch::Tensor &get_rewards() const;

        const torch::Tensor &get_is_done() const;
        const torch::Tensor &get_is_truncated() const;

    private:
        TorchState states;
        torch::Tensor rewards;
        torch::Tensor is_done;
        torch::Tensor is_truncated;
    };
}// namespace arenai::agent

#endif//ARENAI_TORCH_TYPES_H
