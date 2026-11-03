//
// Created by claude on 03/11/2026.
//

#ifndef ARENAI_POTENTIAL_SHAPING_H
#define ARENAI_POTENTIAL_SHAPING_H

#include <vector>

namespace arenai::agent {

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
