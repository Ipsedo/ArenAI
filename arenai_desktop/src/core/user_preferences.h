//
// Created by samuel on 17/07/2026.
//

#ifndef ARENAI_DESKTOP_CORE_USER_PREFERENCES_H
#define ARENAI_DESKTOP_CORE_USER_PREFERENCES_H

#include <filesystem>

#include "../gui/menu.h"

namespace arenai::desktop {

    // Preferences live in the per-user cache directory, never next to the
    // executable: the install folder may be read-only.
    // %LOCALAPPDATA%\ArenAI on Windows, $XDG_CACHE_HOME/arenai
    // (~/.cache/arenai) elsewhere.
    std::filesystem::path preferences_path();

    // returns `defaults` when the file is missing or unreadable; individual
    // fields fall back to `defaults` when absent or invalid
    gui::GameSettings load_preferences(const gui::GameSettings &defaults);

    // best-effort: a failure is logged, never fatal (the cache directory
    // itself may be unwritable)
    void save_preferences(const gui::GameSettings &settings);

}// namespace arenai::desktop

#endif// ARENAI_DESKTOP_CORE_USER_PREFERENCES_H
