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
            const controller::button fire_button(disc_acc[i][0] > disc_acc[i][1]);

            actions.push_back(
                {.left_joystick = joystick_direction,
                 .right_joystick = joystick_canon,
                 .fire_button = fire_button});
        }

        return actions;
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

        auto *vision_ptr = visions_u8.data_ptr<uint8_t>();
        auto *proprioception_ptr = proprioceptions.data_ptr<float>();

        const size_t vision_bytes = static_cast<size_t>(C * H * W) * sizeof(uint8_t);
        constexpr size_t proprioception_bytes = static_cast<size_t>(P) * sizeof(float);

        at::parallel_for(0, N, 1, [&](const int64_t begin, const int64_t end) {
            for (int64_t n = begin; n < end; ++n) {
                std::memcpy(
                    vision_ptr + n * (C * H * W), states[n].vision.pixels.data(), vision_bytes);

                std::memcpy(
                    proprioception_ptr + n * P, states[n].proprioception.data(),
                    proprioception_bytes);
            }
        });

        return {.vision = visions_u8, .proprioception = proprioceptions};
    }

    TorchState
    state_to_tensor(const core::State &state, const int vision_height, const int vision_width) {
        const auto [vision, proprioception] =
            states_to_tensor({state}, vision_height, vision_width);
        return {.vision = vision[0], .proprioception = proprioception[0]};
    }

    TorchStep steps_to_tensor(
        const std::vector<std::tuple<core::State, core::Reward, core::IsDone>> &steps,
        const int vision_height, const int vision_width) {
        std::vector<core::State> states;
        std::vector<torch::Tensor> rewards;
        std::vector<torch::Tensor> are_done;

        for (const auto &[state, reward, is_done]: steps) {
            states.push_back(state);
            rewards.push_back(torch::tensor({reward}, torch::TensorOptions().dtype(torch::kFloat)));
            are_done.push_back(
                torch::tensor({is_done}, torch::TensorOptions().dtype(torch::kBool)));
        }

        return {
            .states = states_to_tensor(states, vision_height, vision_width),
            .rewards = torch::stack(rewards),
            .is_done = torch::stack(are_done)};
    }

}// namespace arenai::agent
