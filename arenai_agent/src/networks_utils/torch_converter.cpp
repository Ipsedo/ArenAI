//
// Created by samuel on 12/10/2025.
//

#include "../networks_utils/torch_converter.h"

#include <ATen/Parallel.h>

#include <arenai_model/constants.h>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    std::vector<core::Action> tensor_to_actions(
        const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions) {
        const auto batch_size = continuous_actions.size(0);

        std::vector<core::Action> actions;
        actions.reserve(batch_size);

        const auto cont_cpu = continuous_actions.cpu();
        const auto disc_cpu = discrete_actions.cpu();
        auto cont_acc = cont_cpu.accessor<float, 2>();
        auto disc_acc = disc_cpu.accessor<float, 2>();

        for (int i = 0; i < batch_size; i++) {
            const controller::joystick joystick_direction{.x = cont_acc[i][0], .y = cont_acc[i][1]};
            const controller::joystick joystick_canon{.x = cont_acc[i][2], .y = cont_acc[i][3]};
            // binary Bernoulli actions: one neuron per discrete action
            const controller::button fire_button(disc_acc[i][0] > 0.5f);
            const controller::button zoom_button(disc_acc[i][1] > 0.5f);

            actions.push_back(
                {.left_joystick = joystick_direction,
                 .right_joystick = joystick_canon,
                 .fire_button = fire_button,
                 .zoom_button = zoom_button});
        }

        return actions;
    }

    const std::vector<std::string> &continuous_action_names() {
        static const std::vector<std::string> names{"dir_x", "dir_y", "canon_x", "canon_y"};
        return names;
    }

    const std::vector<std::string> &discrete_action_names() {
        static const std::vector<std::string> names{"fire", "zoom"};
        return names;
    }

    TorchState states_to_tensor(
        const std::vector<core::State> &states, const int vision_height, const int vision_width) {
        const auto N = static_cast<int64_t>(states.size());
        constexpr int64_t C = 3;
        const int64_t H = vision_height;
        const int64_t W = vision_width;
        constexpr int64_t P = model::ENEMY_PROPRIOCEPTION_SIZE;

        const torch::Tensor visions_u8 = torch::zeros(
            {N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).requires_grad(false));

        const torch::Tensor proprioceptions =
            torch::zeros({N, P}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));

        const torch::Tensor fire_masks =
            torch::zeros({N, 1}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));

        auto *vision_ptr = visions_u8.data_ptr<uint8_t>();
        auto *proprioception_ptr = proprioceptions.data_ptr<float>();
        auto *fire_mask_ptr = fire_masks.data_ptr<float>();

        const size_t vision_bytes = static_cast<size_t>(C * H * W) * sizeof(uint8_t);
        constexpr size_t proprioception_bytes = static_cast<size_t>(P) * sizeof(float);

        at::parallel_for(0, N, 1, [&](const int64_t begin, const int64_t end) {
            for (int64_t n = begin; n < end; ++n) {
                std::memcpy(
                    vision_ptr + n * (C * H * W), states[n].vision.pixels.data(), vision_bytes);

                std::memcpy(
                    proprioception_ptr + n * P, states[n].proprioception.data(),
                    proprioception_bytes);

                fire_mask_ptr[n] = states[n].can_fire ? 1.f : 0.f;
            }
        });

        return {.vision = visions_u8, .proprioception = proprioceptions, .fire_mask = fire_masks};
    }

    TorchState
    state_to_tensor(const core::State &state, const int vision_height, const int vision_width) {
        const auto [vision, proprioception, fire_mask] =
            states_to_tensor({state}, vision_height, vision_width);
        return {
            .vision = vision[0], .proprioception = proprioception[0], .fire_mask = fire_mask[0]};
    }

    torch::Tensor
    discrete_action_mask(const torch::Tensor &fire_mask, const int64_t nb_discrete_actions) {
        // same layout as tensor_to_actions: fire first, the others (zoom) always have an effect
        return torch::cat(
            {fire_mask,
             torch::ones({fire_mask.size(0), nb_discrete_actions - 1}, fire_mask.options())},
            -1);
    }

    TorchStep steps_to_tensor(
        const std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>>
            &steps,
        const int vision_height, const int vision_width) {
        std::vector<core::State> states;
        std::vector<torch::Tensor> rewards;
        std::vector<torch::Tensor> are_done;
        std::vector<torch::Tensor> are_truncated;

        for (const auto &[state, reward, is_done, is_truncated]: steps) {
            states.push_back(state);
            rewards.push_back(torch::tensor({reward}, torch::TensorOptions().dtype(torch::kFloat)));
            are_done.push_back(
                torch::tensor({is_done}, torch::TensorOptions().dtype(torch::kBool)));
            are_truncated.push_back(
                torch::tensor({is_truncated}, torch::TensorOptions().dtype(torch::kBool)));
        }

        return {
            .states = states_to_tensor(states, vision_height, vision_width),
            .rewards = torch::stack(rewards),
            .is_done = torch::stack(are_done),
            .is_truncated = torch::stack(are_truncated)};
    }

}// namespace arenai::agent
