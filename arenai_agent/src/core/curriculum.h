#ifndef ARENAI_AGENT_HOST_SPAWN_CURRICULUM_H
#define ARENAI_AGENT_HOST_SPAWN_CURRICULUM_H

#include <cstdint>
#include <random>

namespace arenai::agent {

    class ProgressCurriculum {
    public:
        ProgressCurriculum(
            float delta_progress, float ratio_low, float ratio_high, int probe_window,
            float boundary_proba, std::uint64_t seed);

        float sample_progress();

        void on_episode_end(int nb_fires, int nb_hits);

        float upper_bound() const;

        bool is_probe() const;

    private:
        void attempt_update();

        float upper;

        float delta_progress;
        float ratio_low;
        float ratio_high;
        int probe_window;
        float boundary_proba;

        bool last_was_probe;
        int nb_probe_episodes;
        long sum_fires;
        long sum_hits;

        std::mt19937 rng;
    };

}// namespace arenai::agent

#endif// ARENAI_AGENT_HOST_SPAWN_CURRICULUM_H
