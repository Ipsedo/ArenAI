#include "./panorama_mask_labels.h"

#include <cmath>
#include <numbers>

namespace arenai::agent {

    namespace {
        // m
        constexpr float MAX_LABEL_DISTANCE = 400.f;

        // column 0 is the leftmost: straight behind, turning right as the index grows
        int column_of(const float bearing, const int columns) {
            const float width = 2.f * std::numbers::pi_v<float> / static_cast<float>(columns);
            const int column =
                static_cast<int>(std::floor(static_cast<float>(columns) / 2.f - bearing / width));
            return (column % columns + columns) % columns;
        }
    }// namespace

    PanoramaMaskLabels
    panorama_mask_labels(const std::vector<EnemySighting> &sightings, const int columns) {
        auto labels = torch::zeros({columns});
        auto ambiguous = torch::zeros({columns});

        auto label = labels.accessor<float, 1>();
        auto doubt = ambiguous.accessor<float, 1>();

        for (const auto &sighting: sightings) {
            const int first = column_of(sighting.bearing + sighting.angular_half_width, columns);
            const int last = column_of(sighting.bearing - sighting.angular_half_width, columns);
            const int nb_covered = ((last - first) % columns + columns) % columns + 1;

            for (int k = 0; k < nb_covered; k++) doubt[(first + k) % columns] = 1.f;

            if (sighting.visible && sighting.distance < MAX_LABEL_DISTANCE)
                label[column_of(sighting.bearing, columns)] = 1.f;
        }

        // a labelled column is never ambiguous
        return {.labels = labels, .weights = 1.f - ambiguous * (1.f - labels)};
    }

}// namespace arenai::agent
