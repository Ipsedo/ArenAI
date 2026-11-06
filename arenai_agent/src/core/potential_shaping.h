//
// Created by samuel on 06/11/2026.
//

#ifndef ARENAI_POTENTIAL_SHAPING_H
#define ARENAI_POTENTIAL_SHAPING_H

#include <vector>

namespace arenai::agent {

    // potential-based shaping (Ng et al. 1999): pays scale * (gamma * phi(s') - phi(s)) at
    // every decision, a terminal state having a zero potential; the optimal policy is left
    // unchanged, the dense gradient toward high potentials is what it adds
    class PotentialShaping {
    public:
        PotentialShaping(float gamma, float scale, int nb_agents);

        void reset(const std::vector<float> &potentials);

        std::vector<float>
        step(const std::vector<float> &next_potentials, const std::vector<bool> &terminal);

    private:
        float gamma;
        float scale;
        std::vector<float> previous;
    };

}// namespace arenai::agent

#endif// ARENAI_POTENTIAL_SHAPING_H
