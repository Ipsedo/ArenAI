//
// Created by samuel on 19/07/2026.
//

#ifndef ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H
#define ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H

#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <arenai_agent/factory.h>

#include "./settings.h"

namespace arenai::desktop {

    agent::AgentAlgorithm to_agent_algorithm(AiAlgorithm algorithm);

    // the config.json a selection uses: the explicitly picked file when set,
    // else the one sitting in the state-dict folder, else in its parent (the
    // training layout: train_NNN/config.json next to train_NNN/save_K/)
    std::optional<std::filesystem::path> resolve_agent_config(const AgentSelection &selection);

    // Dry-run load of the selected agent: builds the networks from the
    // config.json, loads every state dict on CPU and answers one forward pass
    // on a blank state, then throws the agent away. Returns std::nullopt when
    // the selection is a working model, else the exception message to display.
    std::optional<std::string> check_agent(const AgentSelection &selection);

    // the enemy agent ready to play, with the run parameters its training
    // config.json dictates to the game loop
    struct LoadedAgent {
        std::shared_ptr<agent::AbstractAgent> agent;
        int vision_height;
        int vision_width;
        float wanted_frequency;
        int action_repeat;
    };

    // the real load (GPU when `cuda`); throws when the selection is unusable
    LoadedAgent load_agent(const AgentSelection &selection, bool cuda);

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H
