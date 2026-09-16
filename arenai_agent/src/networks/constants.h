//
// Created by samuel on 30/07/2026.
//

#ifndef ARENAI_CONSTANTS_H
#define ARENAI_CONSTANTS_H

namespace arenai::agent {
    constexpr float EPSILON = 1e-8f;

    constexpr float SIGMA_MIN = 1e-4f;
    constexpr float SIGMA_MAX = 1.f;

    // Beta mode/concentration policy: κ ≥ 2 keeps the density unimodal (α, β ≥ 1);
    // the floor sits just above the uniform, the ceiling caps the sharpness (σ ≈ 0.02)
    constexpr float CONCENTRATION_MIN = 2.01f;
    constexpr float CONCENTRATION_MAX = 2000.f;
}// namespace arenai::agent

#endif//ARENAI_CONSTANTS_H
