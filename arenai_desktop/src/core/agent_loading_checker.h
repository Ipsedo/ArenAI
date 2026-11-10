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

    std::optional<std::filesystem::path> resolve_agent_config(const AgentSelection &selection);

    std::optional<std::string> check_agent(const AgentSelection &selection);

    struct LoadedAgent {
        std::shared_ptr<agent::AbstractAgent> agent;
        int vision_height;
        int vision_width;
        float wanted_frequency;
        int action_repeat;
    };

    LoadedAgent load_agent(const AgentSelection &selection, bool cuda);

}// namespace arenai::desktop

#endif//ARENAI_DESKTOP_AGENT_LOADING_CHECKER_H
