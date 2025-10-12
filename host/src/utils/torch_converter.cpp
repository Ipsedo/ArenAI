//
// Created by samuel on 12/10/2025.
//

#include "./torch_converter.h"

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
    std::vector<float> batched_visions;
    batched_visions.reserve(states.size() * 3 * ENEMY_VISION_SIZE * ENEMY_VISION_SIZE);
    std::vector<float> batched_proprioception(states.size() * ENEMY_PROPRIOCEPTION_SIZE);

    for (const auto &[vision, proprioception]: states) {
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < ENEMY_VISION_SIZE; h++)
                for (int w = 0; w < ENEMY_VISION_SIZE; w++)
                    batched_visions.push_back(
                        2.f * static_cast<float>(vision[c][h][w]) / 255.f - 1.f);

        for (int p = 0; p < ENEMY_PROPRIOCEPTION_SIZE; p++)
            batched_proprioception.push_back(proprioception[p]);
    }

    return {
        torch::from_blob(
            batched_visions.data(),
            {static_cast<int>(states.size()), 3, ENEMY_VISION_SIZE, ENEMY_VISION_SIZE},
            torch::TensorOptions().dtype(torch::kFloat)),
        torch::from_blob(
            batched_proprioception.data(),
            {static_cast<int>(states.size()), ENEMY_PROPRIOCEPTION_SIZE},
            torch::TensorOptions().dtype(torch::kFloat))};
}
