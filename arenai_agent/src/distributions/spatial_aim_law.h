#ifndef ARENAI_SPATIAL_AIM_LAW_H
#define ARENAI_SPATIAL_AIM_LAW_H

#include <torch/torch.h>

namespace arenai::agent {

    // the aim point closes the continuous command, after the drive actions
    constexpr int AIM_POINT_SIZE = 2;

    struct AimDraw {
        // [..] int64
        torch::Tensor cell;
        // [.., 2]
        torch::Tensor pre_tanh;
    };

    class SpatialAimLaw {
    public:
        // logits [.., R*C], mu and sigma [.., R*C, 2]
        SpatialAimLaw(
            torch::Tensor logits, torch::Tensor mu, torch::Tensor sigma, int rows, int cols);

        AimDraw sample() const;
        AimDraw mode() const;

        // [.., 2] NDC aim point
        torch::Tensor point(const AimDraw &draw) const;

        // [.., 1] log pi(c) + log p(delta | c)
        torch::Tensor log_prob(const AimDraw &draw) const;

        // [.., 1] H(c)
        torch::Tensor cell_entropy() const;
        // [.., 2] E_c[H(delta_k | c)]
        torch::Tensor offset_entropy() const;

        // [.., 1] KL(this || other)
        torch::Tensor kl(const SpatialAimLaw &other) const;

    private:
        torch::Tensor logits;
        torch::Tensor mu;
        torch::Tensor sigma;
        int rows;
        int cols;

        torch::Tensor log_probas() const;
        // [.., 2] the per-cell tensor read at the drawn cell
        static torch::Tensor at_cell(const torch::Tensor &per_cell, const torch::Tensor &cell);
    };

}// namespace arenai::agent

#endif//ARENAI_SPATIAL_AIM_LAW_H
