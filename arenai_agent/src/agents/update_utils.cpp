//
// Created by claude on 27/10/2026.
//

#include "./update_utils.h"

#include <algorithm>

namespace arenai::agent {
    namespace {
        constexpr float KL_TRIM_FRACTION = 0.01f;
    }// namespace

    float trimmed_kl(const std::vector<torch::Tensor> &kl_per_rows) {
        const auto kl_per_row = torch::cat(kl_per_rows);

        const auto nb_kept = std::max<int64_t>(
            1, static_cast<int64_t>(
                   static_cast<float>(kl_per_row.size(0)) * (1.f - KL_TRIM_FRACTION)));

        // per-row KL is non-negative: sorting ascending puts the outliers past nb_kept
        return std::get<0>(torch::sort(kl_per_row)).slice(0, 0, nb_kept).mean().item<float>();
    }

    std::vector<std::vector<torch::Tensor>> shuffled_batches(
        const torch::Tensor &units, const int64_t batch_size, const int64_t gpu_batch_size) {
        const auto nb_units = units.size(0);
        const auto shuffled = units.index_select(
            0, torch::randperm(
                   nb_units, torch::TensorOptions().dtype(torch::kLong).device(units.device())));

        std::vector<std::vector<torch::Tensor>> batches;
        for (int64_t start = 0; start < nb_units; start += batch_size) {
            const auto batch =
                shuffled.slice(0, start, std::min<int64_t>(start + batch_size, nb_units));

            auto &blocks = batches.emplace_back();
            for (int64_t block = 0; block < batch.size(0); block += gpu_batch_size)
                blocks.push_back(batch.slice(
                    0, block, std::min<int64_t>(block + gpu_batch_size, batch.size(0))));
        }
        return batches;
    }

}// namespace arenai::agent
