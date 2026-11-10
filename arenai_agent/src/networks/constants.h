#ifndef ARENAI_CONSTANTS_H
#define ARENAI_CONSTANTS_H

namespace arenai::agent {
    constexpr float EPSILON = 1e-8f;

    constexpr float LOG_SIGMA_MIN = -5.f;
    constexpr float LOG_SIGMA_MAX = -0.3567f;

    constexpr float LIQUID_TAU_INIT_MIN = 0.1f;
    constexpr float LIQUID_TAU_INIT_MAX = 2.f;

    constexpr float CONCENTRATION_MIN = 2.01f;
    constexpr float CONCENTRATION_MAX = 2000.f;
}// namespace arenai::agent

#endif//ARENAI_CONSTANTS_H
