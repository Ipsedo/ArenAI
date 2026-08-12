//
// Created by samuel on 30/07/2026.
//

#ifndef ARENAI_CONSTANTS_H
#define ARENAI_CONSTANTS_H

namespace arenai::agent {
    constexpr float EPSILON = 1e-8f;

    constexpr float TARGET_SIGMA = 0.3f;
    constexpr float TARGET_FIRE_PROBABILITY = 0.4f;

    constexpr float SIGMA_MIN = 1e-3f;
    constexpr float SIGMA_MAX = 1.f;
}// namespace arenai::agent

#endif//ARENAI_CONSTANTS_H
