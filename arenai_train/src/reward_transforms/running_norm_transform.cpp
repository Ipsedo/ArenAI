//
// Created by samuel on 10/06/2026.
//

#include "./running_norm_transform.h"

using namespace arenai;
using namespace arenai::train;

namespace arenai::train {

    /*
 * Running mean/std
 */
    NormalizedRewardTransform::NormalizedRewardTransform(
        const int memory_size, const float reward_scale)
        : memory_size_(memory_size), write_idx_(0), size_(0), reward_scale_(reward_scale),
          running_sum_(0.0), running_sum_sq_(0.0), reward_history_(memory_size, 0.0) {}

    void NormalizedRewardTransform::on_add(const torch::Tensor &single_step_reward) {
        const auto reward_double = single_step_reward.item<double>();

        if (is_full()) {
            const double old_reward = reward_history_[write_idx_];
            running_sum_ -= old_reward;
            running_sum_sq_ -= old_reward * old_reward;
        }

        reward_history_[write_idx_] = reward_double;
        running_sum_ += reward_double;
        running_sum_sq_ += reward_double * reward_double;

        write_idx_ = (write_idx_ + 1) % memory_size_;
        if (size_ < memory_size_) size_++;
    }

    torch::Tensor NormalizedRewardTransform::transform(const torch::Tensor &batch_step_reward) {

        // transform
        const auto current_size = static_cast<float>(std::max(static_cast<int>(size_), 1));

        const auto reward_mean = static_cast<float>(running_sum_ / current_size);
        const auto reward_var =
            static_cast<float>(running_sum_sq_ / current_size - reward_mean * reward_mean);
        const auto reward_std = std::sqrt(std::max(reward_var, 0.f) + 1e-8f);

        const auto normalized_reward = (batch_step_reward - reward_mean) / reward_std;

        return reward_scale_ * normalized_reward;
    }

    bool NormalizedRewardTransform::is_full() const { return size_ >= memory_size_; }

    /*
 * Non-zero mean
 */

    NormalizedNonZeroTransform::NormalizedNonZeroTransform(const int memory_size)
        : memory_size_(memory_size), write_idx_(0), size_(0), non_zero_nb_(0),
          non_zero_running_sum_sq_(0.0), reward_history_(memory_size, 0.0) {}

    void NormalizedNonZeroTransform::on_add(const torch::Tensor &single_step_reward) {
        const auto reward_double = single_step_reward.item<double>();

        if (is_full()) {
            const double old_reward = reward_history_[write_idx_];

            non_zero_running_sum_sq_ -= old_reward * old_reward;

            non_zero_nb_ -= old_reward != 0.0 ? 1 : 0;
        }

        reward_history_[write_idx_] = reward_double;

        if (reward_double != 0.0) {
            non_zero_running_sum_sq_ += reward_double * reward_double;

            non_zero_nb_ += 1;
        }

        write_idx_ = (write_idx_ + 1) % memory_size_;
        if (size_ < memory_size_) size_++;
    }

    torch::Tensor NormalizedNonZeroTransform::transform(const torch::Tensor &batch_step_reward) {
        const auto current_size = static_cast<float>(std::max(static_cast<int>(non_zero_nb_), 1));
        const auto reward_rms =
            std::sqrt(static_cast<float>(non_zero_running_sum_sq_ / current_size) + 1e-8f);

        return batch_step_reward / reward_rms;
    }

    bool NormalizedNonZeroTransform::is_full() const { return size_ >= memory_size_; }

}// namespace arenai::train
