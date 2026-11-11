#ifndef ARENAI_AGENT_HOST_MISC_H
#define ARENAI_AGENT_HOST_MISC_H

#include <torch/torch.h>

namespace arenai::agent {

    class AbstractFunctionModule : public torch::nn::Module {
    public:
        virtual torch::Tensor forward(const torch::Tensor &input) = 0;

        void pretty_print(std::ostream &stream) const override = 0;
    };

    class Clamp : public AbstractFunctionModule {
    public:
        Clamp(float lower_bound, float upper_bound);

        torch::Tensor forward(const torch::Tensor &x) override;

        void pretty_print(std::ostream &stream) const override;

    private:
        float lower_bound;
        float upper_bound;
    };

    // sigmoid between the bounds: never saturates exactly, the gradient stays alive at both ends
    class SoftClamp : public AbstractFunctionModule {
    public:
        SoftClamp(float lower_bound, float upper_bound);

        torch::Tensor forward(const torch::Tensor &x) override;

        void pretty_print(std::ostream &stream) const override;

        // the pre-activation that forwards to the given value, for the heads' initialisation
        static float inverse(float value, float lower_bound, float upper_bound);

    private:
        float lower_bound;
        float upper_bound;
    };

    class Exp : public AbstractFunctionModule {
    public:
        torch::Tensor forward(const torch::Tensor &x) override;

        void pretty_print(std::ostream &stream) const override;
    };

    // Beta concentration κ, log-scale on the excess κ - 2 bounded between min and max
    class ConcentrationOutput : public AbstractFunctionModule {
    public:
        ConcentrationOutput(float min_concentration, float max_concentration);

        torch::Tensor forward(const torch::Tensor &input) override;

        void pretty_print(std::ostream &stream) const override;

    private:
        float min_log_excess;
        float max_log_excess;
    };

    class SpatialSoftmax : public torch::nn::Module {
    public:
        SpatialSoftmax();

        torch::Tensor forward(const torch::Tensor &input);

        void pretty_print(std::ostream &stream) const override;

    private:
        torch::Tensor log_temperature;
    };

}// namespace arenai::agent

#endif//ARENAI_AGENT_HOST_MISC_H
