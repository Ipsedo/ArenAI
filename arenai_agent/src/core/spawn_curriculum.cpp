//
// Created by samuel on 05/09/2026.
//

#include "./spawn_curriculum.h"

#include <algorithm>

namespace arenai::agent {

    SpawnCurriculum::SpawnCurriculum(
        const float delta_progress, const float ratio_low, const float ratio_high,
        const int probe_window, const float boundary_proba, const std::uint64_t seed)
        : upper(0.f), delta_progress(delta_progress), ratio_low(ratio_low), ratio_high(ratio_high),
          probe_window(probe_window), boundary_proba(boundary_proba), last_was_probe(false),
          nb_probe_episodes(0), sum_fires(0), sum_hits(0), rng(seed) {}

    float SpawnCurriculum::sample_progress() {
        std::uniform_real_distribution unif(0.f, 1.f);

        last_was_probe = unif(rng) < boundary_proba;
        if (last_was_probe) return upper;

        return upper * unif(rng);
    }

    void SpawnCurriculum::on_episode_end(const int nb_fires, const int nb_hits) {
        if (!last_was_probe) return;

        nb_probe_episodes++;
        sum_fires += nb_fires;
        sum_hits += nb_hits;

        if (nb_probe_episodes >= probe_window) attempt_update();
    }

    void SpawnCurriculum::attempt_update() {
        // a window without a single fire counts as failure: an agent that stopped
        // firing must not see its task keep hardening
        const float ratio =
            sum_fires > 0 ? static_cast<float>(sum_hits) / static_cast<float>(sum_fires) : 0.f;

        if (ratio > ratio_high) upper += delta_progress;
        else if (ratio < ratio_low) upper -= delta_progress;

        upper = std::clamp(upper, 0.f, 1.f);

        nb_probe_episodes = 0;
        sum_fires = 0;
        sum_hits = 0;
    }

    float SpawnCurriculum::upper_bound() const { return upper; }

    bool SpawnCurriculum::is_probe() const { return last_was_probe; }

}// namespace arenai::agent
