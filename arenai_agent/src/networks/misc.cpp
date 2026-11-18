#include "./misc.h"

#include <algorithm>
#include <cmath>

#include "./constants.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    torch::Tensor Exp::forward(const torch::Tensor &x) { return torch::exp(x); }

    void Exp::pretty_print(std::ostream &stream) const { stream << name() << "()"; }

    Clamp::Clamp(const float lower_bound, const float upper_bound)
        : lower_bound(lower_bound), upper_bound(upper_bound) {}

    torch::Tensor Clamp::forward(const torch::Tensor &x) {
        return torch::clamp(x, lower_bound, upper_bound);
    }

    void Clamp::pretty_print(std::ostream &stream) const {
        stream << name() << "(min=" << lower_bound << ", max=" << upper_bound << ")";
    }

    SoftClamp::SoftClamp(const float lower_bound, const float upper_bound)
        : SoftClamp(lower_bound, std::vector{upper_bound}) {}

    SoftClamp::SoftClamp(const float lower_bound, const std::vector<float> &upper_bounds)
        : lower_bound(lower_bound), upper_bounds(upper_bounds) {}

    torch::Tensor SoftClamp::forward(const torch::Tensor &x) {
        const auto upper = torch::tensor(upper_bounds, x.options().requires_grad(false));
        return lower_bound + (upper - lower_bound) * torch::sigmoid(x);
    }

    void SoftClamp::pretty_print(std::ostream &stream) const {
        stream << name() << "(min=" << lower_bound << ", max=";
        for (size_t i = 0; i < upper_bounds.size(); i++)
            stream << (i == 0 ? "" : "/") << upper_bounds[i];
        stream << ")";
    }

    float SoftClamp::inverse(const float value, const float lower_bound, const float upper_bound) {
        constexpr float margin = 1e-4f;
        const float ratio =
            std::clamp((value - lower_bound) / (upper_bound - lower_bound), margin, 1.f - margin);
        return std::log(ratio / (1.f - ratio));
    }

    ConcentrationOutput::ConcentrationOutput(
        const float min_concentration, const float max_concentration)
        : min_log_excess(std::log(min_concentration - 2.f)),
          max_log_excess(std::log(max_concentration - 2.f)) {}

    torch::Tensor ConcentrationOutput::forward(const torch::Tensor &input) {
        const auto log_excess =
            min_log_excess + (max_log_excess - min_log_excess) * torch::sigmoid(input);
        return 2.f + torch::exp(log_excess);
    }

    void ConcentrationOutput::pretty_print(std::ostream &stream) const {
        stream << name() << "(min=" << 2.f + std::exp(min_log_excess)
               << ", max=" << 2.f + std::exp(max_log_excess) << ")";
    }

}// namespace arenai::agent
