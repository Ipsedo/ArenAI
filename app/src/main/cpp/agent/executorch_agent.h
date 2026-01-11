//
// Created by samuel on 03/10/2025.
//

#ifndef ARENAI_EXECUTORCH_AGENT_H
#define ARENAI_EXECUTORCH_AGENT_H

#include <random>

#include <android_native_app_glue.h>
#include <executorch/extension/module/module.h>

#include <arenai_core/environment.h>

using namespace executorch;

float phi(const float z);
float theta(const float x);
float theta_inv(const float theta);

float truncated_normal_sample(
    std::mt19937 rng, const float mu, const float sigma, const float min_value,
    const float max_value);

class ExecuTorchAgent {
public:
    ExecuTorchAgent(android_app *app, const std::string &pte_asset_path);

    std::vector<Action> act(const std::vector<State> &state);

private:
    extension::module::Module actor_module;

    std::random_device dev;
    std::mt19937 rng;
};

#endif// ARENAI_EXECUTORCH_AGENT_H
