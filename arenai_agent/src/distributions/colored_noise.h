#ifndef ARENAI_AGENT_HOST_COLORED_NOISE_H
#define ARENAI_AGENT_HOST_COLORED_NOISE_H

#include <torch/torch.h>

namespace arenai::agent {

    // standard normal noise, AR(1) correlated from one draw to the next
    class ColoredNoise {
    public:
        explicit ColoredNoise(float correlation);

        torch::Tensor next(const torch::Tensor &like);

        // ended [N, 1] : those series restart from a fresh draw
        void restart(const torch::Tensor &ended);

        void reset();

    private:
        float correlation;
        torch::Tensor epsilon;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_COLORED_NOISE_H
