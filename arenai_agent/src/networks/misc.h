//
// Created by samuel on 19/05/2026.
//

#ifndef ARENAI_AGENT_HOST_MISC_H
#define ARENAI_AGENT_HOST_MISC_H

#include <torch/torch.h>

namespace arenai::agent {

    class AbstractFunctionModule : public torch::nn::Module {
    public:
        virtual torch::Tensor forward(const torch::Tensor &input) = 0;

        virtual void pretty_print(std::ostream &stream) = 0;
    };

    class Clamp : public AbstractFunctionModule {
    public:
        Clamp(float lower_bound, float upper_bound);

        torch::Tensor forward(const torch::Tensor &x) override;

        void pretty_print(std::ostream &stream) override;

    private:
        float lower_bound;
        float upper_bound;
    };

    class Exp : public AbstractFunctionModule {
    public:
        torch::Tensor forward(const torch::Tensor &x) override;

        void pretty_print(std::ostream &stream) override;
    };

    class SigmaOutput : public AbstractFunctionModule {
    public:
        SigmaOutput(float min_sigma, float max_sigma);

        torch::Tensor forward(const torch::Tensor &input) override;

        void pretty_print(std::ostream &stream) override;

    private:
        float min_log_sigma;
        float max_log_sigma;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_MISC_H
