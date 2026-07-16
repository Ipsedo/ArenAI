//
// Created by samuel on 12/10/2025.
//

#include <ATen/Parallel.h>

#include <arenai_core/constants.h>
#include <arenai_train/torch_converter.h>

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

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
            const controller::joystick joystick_direction{cont_acc[i][0], cont_acc[i][1]};
            const controller::joystick joystick_canon{cont_acc[i][2], cont_acc[i][3]};
            const controller::button fire_button(disc_acc[i][0] > disc_acc[i][1]);

            actions.push_back({joystick_direction, joystick_canon, fire_button});
        }

        return actions;
    }

    std::tuple<torch::Tensor, torch::Tensor> states_to_tensor(
        const std::vector<core::State> &states, const int vision_height, const int vision_width) {
        const auto N = static_cast<int64_t>(states.size());
        constexpr int64_t C = 3;
        const int64_t H = vision_height;
        const int64_t W = vision_width;
        constexpr int64_t P = model::ENEMY_PROPRIOCEPTION_SIZE;

        const torch::Tensor visions_u8 = torch::zeros(
            {N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).requires_grad(false));

        torch::Tensor proprioceptions =
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

        return {visions_u8, proprioceptions};
    }

    std::tuple<torch::Tensor, torch::Tensor>
    state_to_tensor(const core::State &state, const int vision_height, const int vision_width) {
        const auto [vision, proprioception] =
            states_to_tensor({state}, vision_height, vision_width);
        return {vision[0], proprioception[0]};
    }

}// namespace arenai::train
