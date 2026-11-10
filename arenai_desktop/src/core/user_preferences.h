#ifndef ARENAI_DESKTOP_CORE_USER_PREFERENCES_H
#define ARENAI_DESKTOP_CORE_USER_PREFERENCES_H

#include <filesystem>
#include <vector>

#include "./settings.h"

namespace arenai::desktop {

    std::filesystem::path preferences_path();

    GameSettings
    load_preferences(const GameSettings &defaults, const std::vector<AgentPreset> &presets);

    void save_preferences(const GameSettings &settings);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_USER_PREFERENCES_H
