//
// Created by samuel on 22/07/2026.
//

#include "./empty_step_collector.h"

using namespace arenai;
using namespace arenai::train;

EmptyStepCollector::EmptyStepCollector() = default;

void EmptyStepCollector::on_episode_end(const TorchState &final_state) {}
void EmptyStepCollector::on_transition(
    const torch::Tensor &rewards, const torch::Tensor &done, const torch::Tensor &truncated) {}
