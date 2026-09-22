//
// Created by samuel on 05/09/2026.
//

#ifndef ARENAI_AGENT_HOST_SPAWN_CURRICULUM_H
#define ARENAI_AGENT_HOST_SPAWN_CURRICULUM_H

#include <cstdint>
#include <random>

namespace arenai::agent {

    // ADR-style adaptive curriculum on the spawn-zone size. The difficulty is a
    // progress fraction in [0, 1] mapped by the caller onto [initial, final] spawn
    // sizes. Each episode either probes the current upper bound (probability
    // boundary_proba) or draws uniformly below it — the mixture keeps easy episodes
    // in the training data so long-range practice never erases short-range aim.
    // Only probe episodes feed the controller: every probe_window of them, the
    // aggregated hit/fire ratio moves the bound up past ratio_high, down below
    // ratio_low, and not at all in between (hysteresis).
    class SpawnCurriculum {
    public:
        SpawnCurriculum(
            float delta_progress, float ratio_low, float ratio_high, int probe_window,
            float boundary_proba, std::uint64_t seed);

        // draws the difficulty of the next episode, in [0, 1]
        float sample_progress();

        // per-episode totals across every tank; ignored unless the last sampled
        // episode was a probe
        void on_episode_end(int nb_fires, int nb_hits);

        float upper_bound() const;

        // whether the last sampled episode probes the upper bound
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
