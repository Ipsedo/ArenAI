#ifndef ARENAI_Q_FUNCTION_H
#define ARENAI_Q_FUNCTION_H

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "./state_encoder.h"

namespace arenai::agent {

    class QFunction final : public torch::nn::Module {
    public:
        QFunction(
            int encoding_size, int nb_continuous_actions, int nb_discrete_actions,
            const std::vector<int> &hidden_sizes);

        // [N, 2^nb_discrete_actions]
        torch::Tensor q(const torch::Tensor &encoding, const torch::Tensor &continuous_action);

    private:
        torch::nn::Sequential head;
        torch::nn::Linear to_q;
    };

    // clipped double Q: two independent heads on one shared state encoder
    class TwinQNetwork final : public torch::nn::Module {
    public:
        TwinQNetwork(
            const std::shared_ptr<StateEncoder> &encoder, int nb_continuous_actions,
            int nb_discrete_actions, const std::vector<int> &hidden_sizes);

        torch::Tensor encode(const torch::Tensor &vision, const torch::Tensor &sensors);

        std::pair<torch::Tensor, torch::Tensor>
        q_from_encoding(const torch::Tensor &encoding, const torch::Tensor &continuous_action);

        std::pair<torch::Tensor, torch::Tensor>
        q(const torch::Tensor &vision, const torch::Tensor &sensors,
          const torch::Tensor &continuous_action);

    private:
        std::shared_ptr<StateEncoder> encoder;
        std::shared_ptr<QFunction> q1;
        std::shared_ptr<QFunction> q2;
    };

}// namespace arenai::agent

#endif//ARENAI_Q_FUNCTION_H
