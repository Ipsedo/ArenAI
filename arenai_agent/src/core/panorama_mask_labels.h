#ifndef ARENAI_PANORAMA_MASK_LABELS_H
#define ARENAI_PANORAMA_MASK_LABELS_H

#include <vector>

#include <torch/torch.h>

#include "./train_environment.h"

namespace arenai::agent {

    struct PanoramaMaskLabels {
        // [columns]
        torch::Tensor labels;
        torch::Tensor weights;
    };

    PanoramaMaskLabels
    panorama_mask_labels(const std::vector<EnemySighting> &sightings, int columns);

}// namespace arenai::agent

#endif//ARENAI_PANORAMA_MASK_LABELS_H
