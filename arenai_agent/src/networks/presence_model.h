#ifndef ARENAI_PRESENCE_MODEL_H
#define ARENAI_PRESENCE_MODEL_H

#include <vector>

#include <torch/torch.h>

namespace arenai::agent {

    class PresenceModel {
    public:
        virtual ~PresenceModel() = default;

        // [N, rows, cols] logits of an enemy in sight on each cell
        virtual torch::Tensor presence_logits(const torch::Tensor &vision) = 0;

        virtual std::vector<torch::Tensor> presence_parameters() const = 0;
    };

}// namespace arenai::agent

#endif//ARENAI_PRESENCE_MODEL_H
