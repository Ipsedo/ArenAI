//
// Created by claude on 10/11/2026.
//

#include "./torch_types.h"

#include <cstring>

#include <ATen/Parallel.h>

#include <arenai_model/constants.h>

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    namespace {
        constexpr int64_t SHELLS_COLUMN = model::ENEMY_PROPRIOCEPTION_SIZE;
        constexpr int64_t COOLDOWN_COLUMN = model::ENEMY_PROPRIOCEPTION_SIZE + 1;

        std::vector<float> column(const torch::Tensor &state, const int64_t index) {
            const auto values = state.select(-1, index).detach().cpu().contiguous().flatten();
            return {values.data_ptr<float>(), values.data_ptr<float>() + values.numel()};
        }
    }// namespace

    /*
     * TorchState
     */

    TorchState::TorchState(
        const std::vector<core::State> &states, const int vision_height, const int vision_width) {
        const auto N = static_cast<int64_t>(states.size());
        constexpr int64_t C = 3;
        const int64_t H = vision_height;
        const int64_t W = vision_width;
        constexpr int64_t P = model::ENEMY_PROPRIOCEPTION_SIZE;
        constexpr int64_t S = model::ENEMY_STATE_VECTOR_SIZE;

        vision = torch::zeros(
            {N, C, H, W}, torch::TensorOptions().dtype(torch::kUInt8).requires_grad(false));
        state =
            torch::zeros({N, S}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));
        fire_mask =
            torch::zeros({N, 1}, torch::TensorOptions().dtype(torch::kFloat).requires_grad(false));

        auto *vision_ptr = vision.data_ptr<uint8_t>();
        auto *state_ptr = state.data_ptr<float>();
        auto *fire_mask_ptr = fire_mask.data_ptr<float>();

        const size_t vision_bytes = static_cast<size_t>(C * H * W) * sizeof(uint8_t);
        constexpr size_t proprioception_bytes = static_cast<size_t>(P) * sizeof(float);

        for (const auto &s: states)
            TORCH_CHECK(
                static_cast<int64_t>(s.proprioception.size()) == P,
                "TorchState: proprioception of size ", s.proprioception.size(), ", expected ", P);

        at::parallel_for(0, N, 1, [&](const int64_t begin, const int64_t end) {
            for (int64_t n = begin; n < end; ++n) {
                std::memcpy(
                    vision_ptr + n * (C * H * W), states[n].vision.pixels.data(), vision_bytes);

                auto *row = state_ptr + n * S;
                std::memcpy(row, states[n].proprioception.data(), proprioception_bytes);
                row[SHELLS_COLUMN] = states[n].remaining_shells_ratio;
                row[COOLDOWN_COLUMN] = states[n].cooldown_ratio;

                fire_mask_ptr[n] = states[n].can_fire ? 1.f : 0.f;
            }
        });
    }

    TorchState::TorchState(torch::Tensor vision, torch::Tensor state, torch::Tensor fire_mask)
        : vision(std::move(vision)), state(std::move(state)), fire_mask(std::move(fire_mask)) {}

    const torch::Tensor &TorchState::get_vision() const { return vision; }

    const torch::Tensor &TorchState::get_state() const { return state; }

    const torch::Tensor &TorchState::get_fire_mask() const { return fire_mask; }

    std::vector<float> TorchState::get_remaining_shells_ratio() const {
        return column(state, SHELLS_COLUMN);
    }

    std::vector<float> TorchState::get_cooldown_ratio() const {
        return column(state, COOLDOWN_COLUMN);
    }

    torch::Tensor TorchState::get_discrete_action_mask(const int64_t nb_discrete_actions) const {
        return discrete_action_mask(fire_mask, nb_discrete_actions);
    }

    torch::Tensor TorchState::discrete_action_mask(
        const torch::Tensor &fire_mask, const int64_t nb_discrete_actions) {
        // same layout as TorchAction::to_actions: fire first, the others always have an effect
        auto ones_sizes = fire_mask.sizes().vec();
        ones_sizes.back() = nb_discrete_actions - 1;
        return torch::cat({fire_mask, torch::ones(ones_sizes, fire_mask.options())}, -1);
    }

    TorchState TorchState::to(const torch::Device device) const {
        return {vision.to(device), state.to(device), fire_mask.to(device)};
    }

    TorchState TorchState::detach() const {
        return {vision.detach(), state.detach(), fire_mask.detach()};
    }

    TorchState TorchState::slice(const int64_t start, const int64_t end) const {
        return {
            vision.slice(0, start, end), state.slice(0, start, end),
            fire_mask.slice(0, start, end)};
    }

    /*
     * TorchAction
     */

    std::vector<core::Action> TorchAction::to_actions() const {
        const auto batch_size = continuous_action.size(0);

        std::vector<core::Action> actions;
        actions.reserve(batch_size);

        const auto cont_cpu = continuous_action.cpu();
        const auto disc_cpu = discrete_action.cpu();
        auto cont_acc = cont_cpu.accessor<float, 2>();
        auto disc_acc = disc_cpu.accessor<float, 2>();

        for (int i = 0; i < batch_size; i++) {
            const controller::joystick joystick_direction{.x = cont_acc[i][0], .y = cont_acc[i][1]};
            const controller::joystick joystick_canon{.x = cont_acc[i][2], .y = cont_acc[i][3]};
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

    const std::vector<std::string> &TorchAction::continuous_action_names() {
        static const std::vector<std::string> names{"dir_x", "dir_y", "canon_x", "canon_y"};
        return names;
    }

    const std::vector<std::string> &TorchAction::discrete_action_names() {
        static const std::vector<std::string> names{"fire", "zoom"};
        return names;
    }

    /*
     * TorchStep
     */

    TorchStep::TorchStep(
        const std::vector<std::tuple<core::State, core::Reward, core::IsDone, core::IsTruncated>>
            &steps,
        const int vision_height, const int vision_width) {
        std::vector<core::State> env_states;
        std::vector<torch::Tensor> env_rewards;
        std::vector<torch::Tensor> are_done;
        std::vector<torch::Tensor> are_truncated;

        for (const auto &[state, reward, done, truncated]: steps) {
            env_states.push_back(state);
            env_rewards.push_back(
                torch::tensor({reward}, torch::TensorOptions().dtype(torch::kFloat)));
            are_done.push_back(torch::tensor({done}, torch::TensorOptions().dtype(torch::kBool)));
            are_truncated.push_back(
                torch::tensor({truncated}, torch::TensorOptions().dtype(torch::kBool)));
        }

        states = TorchState(env_states, vision_height, vision_width);
        rewards = torch::stack(env_rewards);
        is_done = torch::stack(are_done);
        is_truncated = torch::stack(are_truncated);
    }

    const TorchState &TorchStep::get_states() const { return states; }

    const torch::Tensor &TorchStep::get_rewards() const { return rewards; }

    const torch::Tensor &TorchStep::get_is_done() const { return is_done; }

    const torch::Tensor &TorchStep::get_is_truncated() const { return is_truncated; }

}// namespace arenai::agent
