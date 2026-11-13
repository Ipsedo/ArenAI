#ifndef ARENAI_DESKTOP_CORE_AGENT_VALIDATOR_H
#define ARENAI_DESKTOP_CORE_AGENT_VALIDATOR_H

#include <functional>
#include <optional>
#include <string>

#include "./settings.h"

namespace arenai::desktop {

    using AgentValidator = std::function<std::optional<std::string>(const AgentSelection &)>;

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_AGENT_VALIDATOR_H
