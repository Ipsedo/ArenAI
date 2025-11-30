//
// Created by samuel on 12/10/2025.
//

#include "./torch_converter.h"

#include <ATen/Parallel.h>

#include <arenai_core/constants.h>

std::vector<Action> tensor_to_actions(const torch::Tensor &actions_tensor) {
    if (actions_tensor.sizes().size() != 2)
        throw std::invalid_argument("actions_tensor.sizes().size() != 2");

    const auto batch_size = actions_tensor.size(0);

    std::vector<Action> actions;
    actions.reserve(batch_size);

    for (int i = 0; i < batch_size; i++) {
        const joystick joystick_direction{
            actions_tensor[i][0].item().toFloat(), actions_tensor[i][1].item().toFloat()};
        const joystick joystick_canon{
            actions_tensor[i][2].item().toFloat(), actions_tensor[i][3].item().toFloat()};
        const button fire_button(actions_tensor[i][4].item().toFloat() > 0);

        actions.push_back({joystick_direction, joystick_canon, fire_button});
    }

    return actions;
}

std::tuple<torch::Tensor, torch::Tensor> states_to_tensor(const std::vector<State> &states) {
    const auto N = static_cast<int64_t>(states.size());
    constexpr int64_t C = 3;
    constexpr int64_t H = ENEMY_VISION_SIZE;
    constexpr int64_t W = ENEMY_VISION_SIZE;
    constexpr int64_t P = ENEMY_PROPRIOCEPTION_SIZE;

    const torch::Tensor visions_u8 = torch::zeros(
        {N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).requires_grad(false));

    torch::Tensor proprioceptions =
        torch::zeros({N, P}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));

    auto *vision_ptr = visions_u8.data_ptr<uint8_t>();
    auto *proprioception_ptr = proprioceptions.data_ptr<float>();

    constexpr size_t vision_stride = C * H * W;
    constexpr auto vision_row_bytes = static_cast<size_t>(W);
    constexpr auto proprioception_bytes = static_cast<size_t>(P) * sizeof(float);

    at::parallel_for(0, N, 1, [&](const int64_t begin, const int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            uint8_t *vision_dst = vision_ptr + n * vision_stride;
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t h = 0; h < H; ++h) {
                    const uint8_t *vision_src_row = states[n].vision[c][h].data();
                    std::memcpy(vision_dst, vision_src_row, vision_row_bytes);
                    vision_dst += static_cast<size_t>(W);
                }
            }
            std::memcpy(
                proprioception_ptr + n * P, states[n].proprioception.data(), proprioception_bytes);
        }
    });

    return {visions_u8, proprioceptions};
}
