#include "./spatial_aim_law.h"

#include "./gaussian_tanh.h"

using namespace arenai;
using namespace arenai::agent;

namespace arenai::agent {

    SpatialAimLaw::SpatialAimLaw(
        torch::Tensor logits, torch::Tensor mu, torch::Tensor sigma, const int rows, const int cols)
        : logits(std::move(logits)), mu(std::move(mu)), sigma(std::move(sigma)), rows(rows),
          cols(cols) {}

    torch::Tensor SpatialAimLaw::log_probas() const { return torch::log_softmax(logits, -1); }

    torch::Tensor SpatialAimLaw::at_cell(const torch::Tensor &per_cell, const torch::Tensor &cell) {
        auto index_sizes = per_cell.sizes().vec();
        index_sizes[index_sizes.size() - 2] = 1;
        const auto index = cell.unsqueeze(-1).unsqueeze(-1).expand(index_sizes);
        return per_cell.gather(-2, index).squeeze(-2);
    }

    AimDraw SpatialAimLaw::sample() const {
        const auto probas = torch::softmax(logits, -1);
        const auto flat_cell = torch::multinomial(probas.reshape({-1, probas.size(-1)}), 1);
        const auto cell = flat_cell.reshape(logits.sizes().slice(0, logits.dim() - 1));
        const auto cell_mu = at_cell(mu, cell);
        return {
            .cell = cell, .pre_tanh = cell_mu + at_cell(sigma, cell) * torch::randn_like(cell_mu)};
    }

    AimDraw SpatialAimLaw::mode() const {
        const auto cell = logits.argmax(-1);
        return {.cell = cell, .pre_tanh = at_cell(mu, cell)};
    }

    torch::Tensor SpatialAimLaw::point(const AimDraw &draw) const {
        const auto row = torch::div(draw.cell, cols, "floor").to(torch::kFloat);
        const auto col = draw.cell.remainder(cols).to(torch::kFloat);
        const auto offset = torch::tanh(draw.pre_tanh);

        // NDC: x to the right, y up, image rows going down
        const auto x = (col + 0.5f) * (2.f / static_cast<float>(cols)) - 1.f
                       + offset.select(-1, 0) / static_cast<float>(cols);
        const auto y = 1.f - (row + 0.5f) * (2.f / static_cast<float>(rows))
                       + offset.select(-1, 1) / static_cast<float>(rows);
        return torch::stack({x, y}, -1);
    }

    torch::Tensor SpatialAimLaw::log_prob(const AimDraw &draw) const {
        const auto cell_log_proba = log_probas().gather(-1, draw.cell.unsqueeze(-1));
        const auto offset_log_pdf =
            gaussian_tanh_log_pdf(draw.pre_tanh, at_cell(mu, draw.cell), at_cell(sigma, draw.cell))
                .sum(-1, true);
        return cell_log_proba + offset_log_pdf;
    }

    torch::Tensor SpatialAimLaw::cell_entropy() const {
        const auto log_p = log_probas();
        return -(torch::exp(log_p) * log_p).sum(-1, true);
    }

    torch::Tensor SpatialAimLaw::offset_entropy() const {
        return (torch::softmax(logits, -1).unsqueeze(-1) * gaussian_tanh_entropy(mu, sigma))
            .sum(-2);
    }

    torch::Tensor SpatialAimLaw::kl(const SpatialAimLaw &other) const {
        const auto log_p = log_probas();
        const auto p = torch::exp(log_p);
        const auto cell_kl = (p * (log_p - other.log_probas())).sum(-1, true);
        const auto offset_kl =
            (p.unsqueeze(-1) * gaussian_tanh_kl(mu, sigma, other.mu, other.sigma)).sum({-2, -1});
        return cell_kl + offset_kl.unsqueeze(-1);
    }

}// namespace arenai::agent
