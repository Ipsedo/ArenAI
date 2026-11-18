#include "./colored_noise.h"

#include <cmath>

namespace arenai::agent {

    ColoredNoise::ColoredNoise(const float correlation) : correlation(correlation) {}

    torch::Tensor ColoredNoise::next(const torch::Tensor &like) {
        const auto innovation = torch::randn_like(like);

        if (!epsilon.defined() || epsilon.sizes() != like.sizes()) epsilon = innovation;
        else
            epsilon =
                correlation * epsilon + std::sqrt(1.f - correlation * correlation) * innovation;

        return epsilon;
    }

    void ColoredNoise::restart(const torch::Tensor &ended) {
        if (!epsilon.defined()) return;

        const auto mask = ended.to(epsilon.device()).reshape({-1, 1}).to(torch::kBool);
        epsilon = torch::where(mask, torch::randn_like(epsilon), epsilon);
    }

    void ColoredNoise::reset() { epsilon = torch::Tensor(); }

}// namespace arenai::agent
