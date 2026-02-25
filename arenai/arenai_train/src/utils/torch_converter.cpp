//
// Created by samuel on 12/10/2025.
//

#include "./torch_converter.h"

#include <ATen/Parallel.h>

#include <arenai_core/constants.h>

std::vector<Action>
tensor_to_actions(const torch::Tensor &continuous_actions, const torch::Tensor &discrete_actions) {
    const auto batch_size = continuous_actions.size(0);

    std::vector<Action> actions;
    actions.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        const joystick joystick_direction{
            continuous_actions[i][0].item<float>(), continuous_actions[i][1].item<float>()};
        const joystick joystick_canon{
            continuous_actions[i][2].item<float>(), continuous_actions[i][3].item<float>()};
        const button fire_button(
            discrete_actions[i][0].item<float>() > discrete_actions[i][1].item<float>());

        actions.push_back({joystick_direction, joystick_canon, fire_button});
    }

    return actions;
}

std::tuple<torch::Tensor, torch::Tensor> states_to_tensor(const std::vector<State> &states) {
    const auto N = static_cast<int64_t>(states.size());
    constexpr int64_t C = 3;
    constexpr int64_t H = ENEMY_VISION_HEIGHT;
    constexpr int64_t W = ENEMY_VISION_WIDTH;
    constexpr int64_t P = ENEMY_PROPRIOCEPTION_SIZE;

    const torch::Tensor visions_u8 = torch::zeros(
        {N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).requires_grad(false));

    torch::Tensor proprioceptions =
        torch::zeros({N, P}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));

    auto *vision_ptr = visions_u8.data_ptr<uint8_t>();
    auto *proprioception_ptr = proprioceptions.data_ptr<float>();

    constexpr size_t vision_bytes = static_cast<size_t>(C * H * W) * sizeof(uint8_t);
    constexpr size_t proprioception_bytes = static_cast<size_t>(P) * sizeof(float);

    at::parallel_for(0, N, 1, [&](const int64_t begin, const int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            std::memcpy(vision_ptr + n * (C * H * W), states[n].vision.pixels.data(), vision_bytes);

            std::memcpy(
                proprioception_ptr + n * P, states[n].proprioception.data(), proprioception_bytes);
        }
    });

    return {visions_u8, proprioceptions};
}

std::tuple<torch::Tensor, torch::Tensor> state_to_tensor(const State &state) {
    const auto [vision, proprioception] = states_to_tensor({state});
    return {vision[0], proprioception[0]};
}
