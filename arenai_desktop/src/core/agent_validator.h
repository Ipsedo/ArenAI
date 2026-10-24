//
// Created by samuel on 24/10/2026.
//

#ifndef ARENAI_DESKTOP_CORE_AGENT_VALIDATOR_H
#define ARENAI_DESKTOP_CORE_AGENT_VALIDATOR_H

#include <functional>
#include <optional>
#include <string>

#include "./settings.h"

namespace arenai::desktop {

    // Dry-run check of an agent selection: nullopt = the model loaded and
    // answered a forward pass, else the message to display. The gui depends
    // on this signature only; the real check (agent_loading_checker) pulls
    // arenai_agent in and stays out of the gui hexagon.
    using AgentValidator = std::function<std::optional<std::string>(const AgentSelection &)>;

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_AGENT_VALIDATOR_H
