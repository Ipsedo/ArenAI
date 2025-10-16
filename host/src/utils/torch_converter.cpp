//
// Created by samuel on 12/10/2025.
//

#include "./torch_converter.h"

#include <ATen/Parallel.h>

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

        actions.push_back({joystick_direction, joystick_canon, fire_button, {false}, {false}});
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

    torch::Tensor proprio =
        torch::zeros({N, P}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));

    auto *vis_ptr = visions_u8.data_ptr<uint8_t>();
    auto *prop_ptr = proprio.data_ptr<float>();

    constexpr size_t vision_stride = C * H * W;
    constexpr auto row_bytes = static_cast<size_t>(W);
    constexpr auto prop_bytes = static_cast<size_t>(P) * sizeof(float);

    at::parallel_for(0, N, 1, [&](const int64_t begin, const int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            if (states[n].vision.size() != C)
                throw std::invalid_argument("states[n].vision.size() != C");

            uint8_t *dst = vis_ptr + n * vision_stride;
            for (int64_t c = 0; c < C; ++c) {
                if (states[n].vision[c].size() != H)
                    throw std::invalid_argument("states[n].vision[c].size() != H");

                for (int64_t h = 0; h < H; ++h) {
                    if (states[n].vision[c][h].size() != W)
                        throw std::invalid_argument("states[n].vision[c][h].size() != W");

                    const uint8_t *src_row = states[n].vision[c][h].data();
                    std::memcpy(dst, src_row, row_bytes);
                    dst += static_cast<size_t>(W);
                }
            }
            std::memcpy(prop_ptr + n * P, states[n].proprioception.data(), prop_bytes);
        }
    });

    return {visions_u8.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f), proprio};
}

std::tuple<torch::Tensor, torch::Tensor> state_to_tensor(const State &state) {
    constexpr int64_t C = 3;
    constexpr int64_t H = ENEMY_VISION_SIZE;
    constexpr int64_t W = ENEMY_VISION_SIZE;
    constexpr int64_t P = ENEMY_PROPRIOCEPTION_SIZE;

    const torch::Tensor visions_u8 =
        torch::zeros({C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).requires_grad(false));

    torch::Tensor proprio =
        torch::zeros({P}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));

    auto *vis_ptr = visions_u8.data_ptr<uint8_t>();
    auto *prop_ptr = proprio.data_ptr<float>();

    constexpr auto row_bytes = static_cast<size_t>(W);

    if (state.vision.size() != C) throw std::invalid_argument("states[n].vision.size() != C");

    for (int64_t c = 0; c < C; ++c) {
        if (state.vision[c].size() != H)
            throw std::invalid_argument("states[n].vision[c].size() != H");

        for (int64_t h = 0; h < H; ++h) {
            if (state.vision[c][h].size() != W)
                throw std::invalid_argument("states[n].vision[c][h].size() != W");

            const uint8_t *src_row = state.vision[c][h].data();
            std::memcpy(vis_ptr, src_row, row_bytes);
            vis_ptr += static_cast<size_t>(W);
        }
    }
    std::memcpy(prop_ptr, state.proprioception.data(), static_cast<size_t>(P) * sizeof(float));

    return {visions_u8.to(torch::kFloat).mul_(2.0f / 255.0f).add_(-1.0f), proprio};
}
