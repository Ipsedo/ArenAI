//
// Created by samuel on 03/10/2025.
//

#ifndef PHYVR_EXECUTORCH_AGENT_H
#define PHYVR_EXECUTORCH_AGENT_H

#include <android_native_app_glue.h>
#include <executorch/extension/module/module.h>

#include <phyvr_core/environment.h>

class ExecuTorchAgent {
public:
    ExecuTorchAgent(android_app *app, const std::string &pte_asset_path);

    std::vector<Action> act(const std::vector<State> &state);

private:
    executorch::extension::module::Module actor_module;
};

#endif// PHYVR_EXECUTORCH_AGENT_H
