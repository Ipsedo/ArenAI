//
// Created by samuel on 04/11/2026.
//

#ifndef ARENAI_SHOT_BASELINE_H
#define ARENAI_SHOT_BASELINE_H

namespace arenai::agent {

    class ShotBaseline {
    public:
        explicit ShotBaseline(int window);

        float value() const;

        void update(float sum, int nb_shells);

    private:
        int window;
        long nb_seen;
        float mean;
    };

}// namespace arenai::agent

#endif//ARENAI_SHOT_BASELINE_H
