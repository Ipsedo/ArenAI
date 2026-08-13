//
// Created by samuel on 19/05/2026.
//

#include "./misc.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    /*
     * Exp
     */

    torch::Tensor Exp::forward(const torch::Tensor &x) { return torch::exp(x); }

    void Exp::pretty_print(std::ostream &stream) { stream << name() << "()"; }

    /*
     * Clamp
     */

    Clamp::Clamp(const float lower_bound, const float upper_bound)
        : lower_bound(lower_bound), upper_bound(upper_bound) {}

    torch::Tensor Clamp::forward(const torch::Tensor &x) {
        return torch::clamp(x, lower_bound, upper_bound);
    }

    void Clamp::pretty_print(std::ostream &stream) {
        stream << name() << "(min=" << lower_bound << ", max=" << upper_bound << ")";
    }

    /*
     * Sigma of normal distribution output layer
     */

    SigmaOutput::SigmaOutput(const float min_sigma, const float max_sigma)
        : min_log_sigma(std::log(min_sigma)), max_log_sigma(std::log(max_sigma)) {}

    torch::Tensor SigmaOutput::forward(const torch::Tensor &input) {
        const auto log_sigma =
            min_log_sigma + (max_log_sigma - min_log_sigma) * torch::sigmoid(input);
        return torch::exp(log_sigma);
    }

    void SigmaOutput::pretty_print(std::ostream &stream) {
        stream << name() << "(min=" << std::exp(min_log_sigma)
               << ", max=" << std::exp(max_log_sigma) << ")";
    }

}// namespace arenai::agent
