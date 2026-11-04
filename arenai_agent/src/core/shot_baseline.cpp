//
// Created by samuel on 04/11/2026.
//

#include "./shot_baseline.h"

#include <cmath>

namespace arenai::agent {

    ShotBaseline::ShotBaseline(const int window) : window(window), nb_seen(0), mean(0.f) {}

    float ShotBaseline::value() const { return mean; }

    void ShotBaseline::update(const float sum, const int nb_shells) {
        if (nb_shells <= 0) return;

        const float batch_mean = sum / static_cast<float>(nb_shells);

        if (nb_seen < window) {
            const auto total = static_cast<float>(nb_seen + nb_shells);
            mean += (sum - static_cast<float>(nb_shells) * mean) / total;
        } else {
            const float kept =
                std::pow(1.f - 1.f / static_cast<float>(window), static_cast<float>(nb_shells));
            mean += (1.f - kept) * (batch_mean - mean);
        }

        nb_seen += nb_shells;
    }

}// namespace arenai::agent
