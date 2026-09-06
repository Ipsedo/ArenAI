//
// Created by samuel on 19/07/2026.
//

#ifndef ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H
#define ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H

#include <filesystem>
#include <optional>
#include <string>

#include <arenai_agent/factory.h>

#include "../gui/menu.h"

namespace arenai::desktop {

    agent::AgentAlgorithm to_agent_algorithm(gui::AiAlgorithm algorithm);

    // the config.json a selection uses: the explicitly picked file when set,
    // else the one sitting in the state-dict folder, else in its parent (the
    // training layout: train_NNN/config.json next to train_NNN/save_K/)
    std::optional<std::filesystem::path> resolve_agent_config(const gui::AgentSelection &selection);

    // Dry-run load of the selected agent: builds the networks from the
    // config.json, loads every state dict on CPU and answers one forward pass
    // on a blank state, then throws the agent away. Returns std::nullopt when
    // the selection is a working model, else the exception message to display.
    std::optional<std::string> check_agent(const gui::AgentSelection &selection);

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H
