#ifndef ARENAI_DESKTOP_CORE_AGENT_PRESETS_H
#define ARENAI_DESKTOP_CORE_AGENT_PRESETS_H

#include <filesystem>
#include <vector>

#include "./settings.h"

namespace arenai::desktop {

    std::vector<AgentPreset> agent_presets(const std::filesystem::path &resources_folder);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_AGENT_PRESETS_H
