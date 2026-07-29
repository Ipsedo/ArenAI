//
// Created by samuel on 19/07/2026.
//

#ifndef ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H
#define ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H

#include <filesystem>
#include <optional>
#include <string>

#include "../game.h"

namespace arenai::desktop {

    // Dry-run load of the SAC agent: builds the networks and loads every
    // state dict on CPU, then throws the agent away. Returns std::nullopt
    // when the folder is a valid checkpoint, else the message to display.
    std::optional<std::string>
    check_agent_folder(const ModelOptions &model_options, const std::filesystem::path &folder);

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H
