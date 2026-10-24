//
// Created by samuel on 24/10/2026.
//

#ifndef ARENAI_DESKTOP_CORE_AGENT_PRESETS_H
#define ARENAI_DESKTOP_CORE_AGENT_PRESETS_H

#include <filesystem>
#include <vector>

#include "./settings.h"

namespace arenai::desktop {

    // The trained models shipped in resources/trained_models, hardcoded in
    // the order the menu lists them: the first one is the default
    // selection. Paths come back absolute, resolved against the
    // resources folder found next to the executable.
    std::vector<AgentPreset> agent_presets(const std::filesystem::path &resources_folder);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_AGENT_PRESETS_H
