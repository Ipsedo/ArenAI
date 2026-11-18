#include <algorithm>

#include "./curriculum.h"

namespace arenai::agent {

    ProgressCurriculum::ProgressCurriculum(
        const float delta_progress, const float ratio_low, const float ratio_high,
        const int probe_window, const float boundary_proba, const std::uint64_t seed)
        : upper(0.f), delta_progress(delta_progress), ratio_low(ratio_low), ratio_high(ratio_high),
          probe_window(probe_window), boundary_proba(boundary_proba), last_was_probe(false),
          nb_probe_episodes(0), sum_kills(0), sum_tanks(0), rng(seed) {}

    float ProgressCurriculum::sample_progress() {
        std::uniform_real_distribution unif(0.f, 1.f);

        last_was_probe = unif(rng) < boundary_proba;
        if (last_was_probe) return upper;

        return upper * unif(rng);
    }

    void ProgressCurriculum::on_episode_end(const int nb_tanks, const int nb_kills) {
        if (!last_was_probe) return;

        nb_probe_episodes++;
        sum_kills += nb_kills;
        sum_tanks += nb_tanks;

        if (nb_probe_episodes >= probe_window) attempt_update();
    }

    void ProgressCurriculum::attempt_update() {
        const float ratio =
            sum_tanks > 0 ? static_cast<float>(sum_kills) / static_cast<float>(sum_tanks) : 0.f;

        if (ratio > ratio_high) upper += delta_progress;
        else if (ratio < ratio_low) upper -= delta_progress;

        upper = std::clamp(upper, 0.f, 1.f);

        nb_probe_episodes = 0;
        sum_kills = 0;
        sum_tanks = 0;
    }

    float ProgressCurriculum::upper_bound() const { return upper; }

    bool ProgressCurriculum::is_probe() const { return last_was_probe; }

}// namespace arenai::agent
