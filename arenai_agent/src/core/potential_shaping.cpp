//
// Created by claude on 03/11/2026.
//

#include "./potential_shaping.h"

namespace arenai::agent {

    PotentialShaping::PotentialShaping(const float gamma, const float scale, const int nb_agents)
        : gamma(gamma), scale(scale), previous(nb_agents, 0.f) {}

    void PotentialShaping::reset(const std::vector<float> &potentials) { previous = potentials; }

    std::vector<float> PotentialShaping::step(
        const std::vector<float> &next_potentials, const std::vector<bool> &terminal) {
        std::vector<float> rewards(previous.size(), 0.f);

        for (size_t i = 0; i < previous.size(); i++) {
            const float next = terminal[i] ? 0.f : next_potentials[i];

            rewards[i] = scale * (gamma * next - previous[i]);
            previous[i] = next;
        }

        return rewards;
    }

}// namespace arenai::agent
