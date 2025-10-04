//
// Created by samuel on 03/10/2025.
//

#include "./executorch_agent.h"
#include "./loader.h"

ExecuTorchAgent::ExecuTorchAgent(android_app *app,
                                 const std::string &pte_asset_path)
    : actor_module(copy_asset_to_files(app->activity->assetManager,
                                       pte_asset_path.c_str(),
                                       get_cache_dir(app))) {}

std::vector<Action> ExecuTorchAgent::act(const std::vector<State> &state) {
  return {};
}
