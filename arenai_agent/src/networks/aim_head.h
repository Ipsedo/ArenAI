#ifndef ARENAI_AIM_HEAD_H
#define ARENAI_AIM_HEAD_H

#include <torch/torch.h>

#include "../distributions/spatial_aim_law.h"
#include "./misc.h"

namespace arenai::agent {

    class AimHead : public torch::nn::Module {
    public:
        AimHead(int map_channels, int state_size, float initial_sigma);

        // map [N, C, R, C'], state [N, S]
        SpatialAimLaw forward(const torch::Tensor &map, const torch::Tensor &state);

    private:
        torch::nn::Conv2d cells{nullptr};
        std::shared_ptr<SoftClamp> log_sigma;
    };

}// namespace arenai::agent

#endif//ARENAI_AIM_HEAD_H
