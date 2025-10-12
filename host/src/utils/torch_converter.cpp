//
// Created by samuel on 12/10/2025.
//

#include "./torch_converter.h"

#include <ATen/Parallel.h>

std::vector<Action> actions_tensor_to_core(const torch::Tensor &actions_tensor) {
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

        actions.push_back({joystick_direction, joystick_canon, fire_button, {false}, {false}});
    }

    return actions;
}

std::tuple<torch::Tensor, torch::Tensor> state_core_to_tensor(const std::vector<State> &states) {
    const int64_t N = static_cast<int64_t>(states.size());
    constexpr int64_t C = 3;
    constexpr int64_t H = ENEMY_VISION_SIZE;
    constexpr int64_t W = ENEMY_VISION_SIZE;
    constexpr int64_t P = ENEMY_PROPRIOCEPTION_SIZE;

    const torch::Tensor visions_u8 =
        torch::empty({N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8));

    torch::Tensor proprio = torch::empty({N, P}, torch::TensorOptions().dtype(torch::kFloat32));

    auto *vis_ptr = visions_u8.data_ptr<uint8_t>();
    auto *prop_ptr = proprio.data_ptr<float>();

    constexpr int64_t vision_stride = C * H * W;// éléments (uint8)

    at::parallel_for(0, N, 1, [&](const int64_t begin, const int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            const auto &[vision, proprioception] = states[n];

            uint8_t *dst = vis_ptr + n * vision_stride;
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    const uint8_t *src_row = vision[c][h].data();
                    std::memcpy(dst, src_row, static_cast<size_t>(W));
                    dst += W;
                }
            }

            std::memcpy(
                prop_ptr + n * P, proprioception.data(), static_cast<size_t>(P) * sizeof(float));
        }
    });

    return {visions_u8.to(torch::kFloat32).mul_(2.0f / 255.0f).add_(-1.0f), proprio};
}
